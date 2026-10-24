"""
Slide-level report pipeline: five steps, each with one job, and one of them an LLM.

  1. ClassifierAgent   CNN labels every cell image and gives a confidence.
  2. QCAgent           rejects low-confidence cells; refuses slides with too few
                       accepted white cells (a manual differential counts 100).
  3. DifferentialAgent SQL turns accepted cells into a % differential with
                       reference-range flags (serve/sql/analytics.sql).
  4. ReportAgent       writes a short draft report FROM THOSE NUMBERS ONLY.
                       Uses Claude when ANTHROPIC_API_KEY (or another Anthropic
                       credential) is configured, otherwise a fixed template.
  5. VerifierAgent     checks the draft: every percentage must match the SQL
                       table, no disease names, the review disclaimer present.
                       A failed draft is blocked and replaced by the template.

The LLM never sees images and never decides anything on its own; it only
words numbers that deterministic code computed and checked. Every report is a
draft for a qualified hematologist to sign off.
"""

import json
import os
import re

from data import CLASSES

CONFIDENCE_GATE = float(os.environ.get("CONFIDENCE_GATE", "0.70"))
MIN_WBC = int(os.environ.get("MIN_WBC", "100"))
DISCLAIMER = "DRAFT - automated analysis; requires review and sign-off by a qualified hematologist."
FORBIDDEN = ["leukemia", "leukaemia", "lymphoma", "cancer", "malignan", "infection", "sepsis",
             "diagnosis of", "diagnosed", "confirms", "confirmed", "parasite", "allergy", "allergic"]
NICE = {"immature_granulocyte": "immature granulocytes", "erythroblast": "nucleated red cells (erythroblasts)"}


# ----------------------------------------------------------------- 1. classifier
class ClassifierAgent:
    def __init__(self, classifier):
        self.clf = classifier

    def run(self, x_uint8):
        """(B, 3, 64, 64) uint8 -> list of (class, confidence, probs)."""
        probs = self.clf.predict_tensor(x_uint8)
        return [(CLASSES[int(p.argmax())], float(p.max()), p.tolist()) for p in probs]


# ----------------------------------------------------------------- 2. quality control
class QCAgent:
    def accept_cell(self, confidence):
        return confidence >= CONFIDENCE_GATE

    def check_slide(self, qc_row):
        """qc_row from the slide_qc query -> (enough_cells, warnings)."""
        if qc_row is None or qc_row["wbc_accepted"] < MIN_WBC:
            n = 0 if qc_row is None else qc_row["wbc_accepted"]
            return False, [f"only {n} white cells passed the confidence gate; at least {MIN_WBC} are needed"]
        warnings = []
        if qc_row["rejected_fraction"] > 0.10:
            warnings.append(f"{qc_row['rejected_fraction'] * 100:.0f} in 100 cells were rejected as "
                            "low-confidence; slide quality should be checked")
        return True, warnings


# ----------------------------------------------------------------- 3. differential
class DifferentialAgent:
    def __init__(self, store):
        self.store = store

    def run(self, slide_id):
        return self.store.query("differential", slide_id=slide_id)


# ----------------------------------------------------------------- 4. report writer
def template_report(slide_id, table, qc_notes):
    wbc = table[0]["wbc_counted"] if table else 0
    lines = [f"Slide {slide_id}: differential of {wbc} white cells."]
    for r in sorted(table, key=lambda r: -r["pct"]):
        if r["cell_type"] == "erythroblast":
            lines.append(f"- {NICE['erythroblast']}: {r['pct']} per 100 white cells ({r['flag']}).")
        else:
            name = NICE.get(r["cell_type"], r["cell_type"] + "s")
            lines.append(f"- {name}: {r['pct']}% (reference {r['low_pct']}-{r['high_pct']}%, {r['flag']}).")
    flagged = [r for r in table if r["flag"] != "normal"]
    if flagged:
        lines.append("Outside reference range: " + ", ".join(
            f"{NICE.get(r['cell_type'], r['cell_type'])} {r['flag']}" for r in flagged) + ".")
    else:
        lines.append("All counted cell types are within reference ranges.")
    for note in qc_notes:
        lines.append(f"Quality note: {note}.")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You draft short white-cell differential summaries for a hematology lab.
You receive a table computed by the lab's software. Write 4-8 plain sentences that:
- state how many white cells were counted;
- give every cell type's percentage exactly as written in the table (copy the numbers; do not round or recompute);
- say which types are above or below the reference range, using only the words "above" or "below the reference range";
- end with this exact sentence: "{DISCLAIMER}"
Do not name any disease or cause, do not suggest a diagnosis or treatment, and do not add numbers that are not in the table."""


class ReportAgent:
    def __init__(self, model="claude-opus-5-5"):
        self.model = model
        self.client = None
        try:
            import anthropic
            if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") \
                    or os.environ.get("ANTHROPIC_PROFILE"):
                self.client = anthropic.Anthropic()
        except ImportError:
            pass

    @property
    def mode(self):
        return "llm" if self.client else "template"

    def run(self, slide_id, table, qc_notes):
        if not self.client:
            return "template", template_report(slide_id, table, qc_notes)
        payload = json.dumps({"slide_id": slide_id, "differential": table, "quality_notes": qc_notes}, indent=1)
        try:
            # server-side fallback: if Opus declines the request, the API retries on a fallback model
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": f"Differential table:\n{payload}"}],
            )
        except Exception as e:                       # network / auth / rate limit: never block the lab
            return "template", template_report(slide_id, table, qc_notes + [f"LLM unavailable ({type(e).__name__})"])
        if resp.stop_reason == "refusal":
            return "template", template_report(slide_id, table, qc_notes)
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        return "llm", text


# ----------------------------------------------------------------- 5. verifier
class VerifierAgent:
    def run(self, text, table):
        """-> (passed, notes). Every percentage in the text must equal one in the table."""
        notes = []
        allowed = {f"{r['pct']:.1f}" for r in table} | {f"{r['pct']:g}" for r in table}
        allowed |= {f"{v:g}" for r in table for v in (r["low_pct"], r["high_pct"]) if v is not None}
        allowed |= {str(r["wbc_counted"]) for r in table if r["wbc_counted"] is not None}
        for num in re.findall(r"(\d+(?:\.\d+)?)\s*%", text):
            if num not in allowed and f"{float(num):g}" not in allowed:
                notes.append(f"percentage {num}% is not in the computed table")
        low = text.lower()
        for word in FORBIDDEN:
            if word in low:
                notes.append(f"forbidden term: '{word}'")
        if DISCLAIMER not in text:
            notes.append("review disclaimer missing")
        for r in table:                               # every out-of-range type must be mentioned
            if r["flag"] != "normal":
                key = r["cell_type"].split("_")[-1].rstrip("s")[:6]
                if key not in low:
                    notes.append(f"out-of-range type not mentioned: {r['cell_type']}")
        return not notes, notes


# ----------------------------------------------------------------- orchestration
class SlidePipeline:
    def __init__(self, classifier, store, report_agent=None):
        self.classify = ClassifierAgent(classifier)
        self.qc = QCAgent()
        self.diff = DifferentialAgent(store)
        self.writer = report_agent or ReportAgent()
        self.verify = VerifierAgent()
        self.store = store
        self.version = classifier.version

    def add_cells(self, slide_id, x_uint8, true_classes=None, source="upload"):
        self.store.ensure_slide(slide_id, source)
        results = self.classify.run(x_uint8)
        for i, (cls, conf, probs) in enumerate(results):
            self.store.log_cell(slide_id, self.version, cls, conf, probs, self.qc.accept_cell(conf),
                                true_classes[i] if true_classes else None)
        return results

    def report(self, slide_id):
        qc_rows = self.store.query("slide_qc", slide_id=slide_id)
        qc_row = qc_rows[0] if qc_rows else None
        enough, qc_notes = self.qc.check_slide(qc_row)
        if not enough:
            text = f"Slide {slide_id}: not reported. {qc_notes[0].capitalize()}.\n{DISCLAIMER}"
            rid = self.store.save_report(slide_id, "template", "insufficient_cells", text, [])
            return {"report_id": rid, "status": "insufficient_cells", "author": "template", "text": text,
                    "verifier_notes": [], "differential": [], "qc": qc_row}
        table = self.diff.run(slide_id)
        author, text = self.writer.run(slide_id, table, qc_notes)
        passed, notes = self.verify.run(text, table)
        status = "draft_for_review"
        if not passed:                                # block the LLM draft, fall back to the checked template
            status = "blocked_llm_draft_replaced"
            author, text = "template", template_report(slide_id, table, qc_notes)
            passed2, notes2 = self.verify.run(text, table)
            notes = notes + ([] if passed2 else ["template also failed: " + "; ".join(notes2)])
        rid = self.store.save_report(slide_id, author, status, text, notes)
        return {"report_id": rid, "status": status, "author": author, "text": text,
                "verifier_notes": notes, "differential": table, "qc": qc_row}
