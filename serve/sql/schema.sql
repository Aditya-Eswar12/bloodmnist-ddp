-- Prediction log for the blood-cell classifier.
-- A "slide" is one blood smear; the classifier sees many single-cell images per slide.

CREATE TABLE IF NOT EXISTS slides (
    slide_id    TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    source      TEXT,                 -- e.g. 'simulated', 'upload'
    note        TEXT
);

CREATE TABLE IF NOT EXISTS cells (
    cell_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    slide_id      TEXT NOT NULL REFERENCES slides(slide_id),
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    model_version TEXT NOT NULL,
    pred_class    TEXT NOT NULL,
    confidence    REAL NOT NULL,      -- highest softmax probability
    probs         TEXT NOT NULL,      -- JSON list of 8 probabilities
    accepted      INTEGER NOT NULL,   -- 1 if it passed the confidence gate
    true_class    TEXT                -- known only for simulated data or after expert review
);
CREATE INDEX IF NOT EXISTS idx_cells_slide ON cells(slide_id);

CREATE TABLE IF NOT EXISTS reports (
    report_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    slide_id      TEXT NOT NULL REFERENCES slides(slide_id),
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    author        TEXT NOT NULL,      -- 'llm' or 'template'
    status        TEXT NOT NULL,      -- 'draft_for_review', 'blocked', 'insufficient_cells'
    text          TEXT NOT NULL,
    verifier_notes TEXT
);

-- Approximate adult reference ranges, % of white cells (textbook values; not for clinical use).
-- Erythroblasts (nucleated red cells) are counted per 100 white cells; platelets are not white cells.
-- Immature granulocytes and erythroblasts have a reporting threshold instead of 0: with a classifier
-- that is ~98% accurate, a handful of misclassified cells per slide would otherwise flag every slide.
-- (Automated analysers commonly flag immature granulocytes above roughly 2%.)
CREATE TABLE IF NOT EXISTS reference_ranges (
    cell_type TEXT PRIMARY KEY,
    low_pct   REAL NOT NULL,
    high_pct  REAL NOT NULL
);
INSERT OR IGNORE INTO reference_ranges VALUES
    ('neutrophil',            40.0, 70.0),
    ('lymphocyte',            20.0, 40.0),
    ('monocyte',               2.0,  8.0),
    ('eosinophil',             1.0,  4.0),
    ('basophil',               0.0,  1.0),
    ('immature_granulocyte',   0.0,  2.0),
    ('erythroblast',           0.0,  1.0);
