"""
Conditional DCGAN for BloodMNIST (64x64 RGB, 8 classes).

Based on the team's draft `cgan_bloodmnist.py`, made to share this project's
data loading and to save what the evaluation needs.

Two networks play a game:
  Generator     G(z, y): random noise z + a class label y  -> a fake 64x64 cell image
  Discriminator D(x, y): an image + the class it claims to be -> "real" or "fake" score

D learns to tell real from fake; G learns to fool D. "Conditional" means both
networks see the label, so G learns one distribution per class and we can ask
it for, say, 1,000 monocytes.

Two versions:
  --arch dcgan  the first version (BCE loss, BatchNorm discriminator, label as an extra
                channel). It mode-collapsed: every sample of a class looked the same.
  --arch sngan  (default) spectral-normalised discriminator with a projection head,
                hinge loss, DiffAugment on real and fake images, and a mode-seeking term
                that rewards different images for different noise vectors.

Run:  python src/cgan.py --epochs 25          -> checkpoints/cgan.pt, results/cgan_samples_epXXX.png
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from data import load_split, normalise, dihedral, CLASSES

N_CLASSES, IMG, CH = 8, 64, 3


class Generator(nn.Module):
    """Label -> learned embedding, concatenated to the noise vector, then four
    transposed convolutions that double the size each time: 4 -> 8 -> 16 -> 32 -> 64."""

    def __init__(self, z_dim=100, embed_dim=50, ngf=32):
        super().__init__()
        self.z_dim = z_dim
        self.label_embed = nn.Embedding(N_CLASSES, embed_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim + embed_dim, ngf * 8, 4, 1, 0, bias=False),   # 1 -> 4
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),             # 4 -> 8
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),             # 8 -> 16
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),                 # 16 -> 32
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, CH, 4, 2, 1, bias=False),                      # 32 -> 64
            nn.Tanh(),                                                              # pixels in [-1, 1]
        )

    def forward(self, z, y):
        h = torch.cat([z, self.label_embed(y)], dim=1)
        return self.net(h[:, :, None, None])


class Discriminator(nn.Module):
    """The label becomes a learned 64x64 map stacked onto the RGB image as a 4th
    channel, so D judges both 'is it realistic?' and 'is it this class?'."""

    def __init__(self, ndf=32):
        super().__init__()
        self.label_embed = nn.Embedding(N_CLASSES, IMG * IMG)
        self.net = nn.Sequential(
            nn.Conv2d(CH + 1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),              # 64 -> 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),                                         # 4 -> 1
        )

    def forward(self, x, y):
        lm = self.label_embed(y).view(-1, 1, IMG, IMG)
        return self.net(torch.cat([x, lm], dim=1)).view(-1)       # one logit per image


class SNDiscriminator(nn.Module):
    """Spectrally normalised discriminator with a projection head (Miyato et al., 2018).

    Spectral normalisation divides every weight matrix by its largest singular value,
    so no layer can amplify its input by more than 1. That keeps D smooth (Lipschitz)
    and its gradients to G informative, which is what the plain DCGAN lacked.
    Projection: score = w . h(x) + embed(y) . h(x), i.e. "realistic?" plus
    "does it match class y?" read off the same features h(x)."""

    def __init__(self, ndf=32):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.body = nn.Sequential(
            sn(nn.Conv2d(CH, ndf, 4, 2, 1)), nn.LeakyReLU(0.1, True),             # 64 -> 32
            sn(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)), nn.LeakyReLU(0.1, True),        # 32 -> 16
            sn(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)), nn.LeakyReLU(0.1, True),    # 16 -> 8
            sn(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)), nn.LeakyReLU(0.1, True),    # 8 -> 4
        )
        self.linear = sn(nn.Linear(ndf * 8, 1))
        self.embed = sn(nn.Embedding(N_CLASSES, ndf * 8))

    def forward(self, x, y):
        h = self.body(x).sum(dim=(2, 3))                                           # (B, 8*ndf)
        return self.linear(h).view(-1) + (self.embed(y) * h).sum(dim=1)


def diff_augment(t, max_shift=4):
    """DiffAugment (Zhao et al., 2020): the same random, differentiable changes applied to
    real AND generated images before the discriminator sees them, so D cannot win by
    memorising the training images. Brightness, saturation, contrast, a small shift, a
    90-degree rotation and a cut-out square."""
    B = t.size(0)
    t = t + (torch.rand(B, 1, 1, 1) - 0.5) * 0.4
    m = t.mean(dim=1, keepdim=True)
    t = (t - m) * (torch.rand(B, 1, 1, 1) * 2) + m
    m = t.mean(dim=(1, 2, 3), keepdim=True)
    t = (t - m) * (torch.rand(B, 1, 1, 1) + 0.5) + m
    sh = torch.randint(-max_shift, max_shift + 1, (2,))
    t = torch.roll(t, shifts=(int(sh[0]), int(sh[1])), dims=(2, 3))
    t = torch.rot90(t, int(torch.randint(4, (1,))), dims=(2, 3))
    cy, cx = torch.randint(0, IMG, (2,))
    mask = torch.ones_like(t[:, :1])
    mask[:, :, max(0, cy - 16):cy + 16, max(0, cx - 16):cx + 16] = 0
    return t * mask


def dcgan_init(m):
    """DCGAN paper: weights from N(0, 0.02). Large initial weights let D win at once."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


@torch.no_grad()
def diversity(G, cls, n=64):
    """Mean pairwise L2 distance among n samples of one class. Mode collapse
    (G ignores z and always draws the same cell) drives this towards 0."""
    G.eval()
    imgs = G(torch.randn(n, G.z_dim), torch.full((n,), cls)).view(n, -1)
    G.train()
    d = torch.cdist(imgs, imgs)
    iu = torch.triu_indices(n, n, 1)
    return d[iu[0], iu[1]].mean().item()


@torch.no_grad()
def save_grid(G, path, per_class=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G.eval()
    fig, ax = plt.subplots(N_CLASSES, per_class, figsize=(per_class * 0.9, N_CLASSES * 0.95))
    for c in range(N_CLASSES):
        imgs = (G(torch.randn(per_class, G.z_dim), torch.full((per_class,), c)) + 1) / 2
        for j in range(per_class):
            ax[c, j].imshow(imgs[j].permute(1, 2, 0).clamp(0, 1).numpy())
            ax[c, j].axis("off")
        ax[c, 0].set_title(CLASSES[c], fontsize=6, loc="left")
    plt.tight_layout(pad=0.2)
    plt.savefig(path, dpi=110)
    plt.close()
    G.train()


@torch.no_grad()
def synthesise(G, cls, n, bs=256):
    """n fake images of one class as uint8 (n, 3, 64, 64), same format as load_split."""
    G.eval()
    out = []
    for i in range(0, n, bs):
        b = min(bs, n - i)
        img = G(torch.randn(b, G.z_dim), torch.full((b,), cls))
        out.append(((img + 1) * 127.5).round().clamp(0, 255).to(torch.uint8))
    return torch.cat(out)


def train(args):
    torch.manual_seed(args.seed)
    x, y = load_split("train")
    sngan = args.arch == "sngan"
    G = Generator(ngf=args.ngf)
    G.apply(dcgan_init)
    if sngan:
        D = SNDiscriminator(ndf=32)
        optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
        optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))
    else:
        # The first version. It mode-collapsed (diversity ~11 vs ~40 for real cells); kept for the record.
        D = Discriminator(ndf=args.ngf)
        D.apply(dcgan_init)
        # beta1 = 0.5 (not 0.9): less momentum, because the opponent keeps changing
        optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
        optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    bs = args.batch
    history = []
    for ep in range(1, args.epochs + 1):
        t0, perm, dsum, gsum, msum, nb = time.time(), torch.randperm(len(x)), 0.0, 0.0, 0.0, 0
        for i in range(0, len(x) - bs + 1, bs):
            idx = perm[i:i + bs]
            if sngan:
                real, yr = normalise(x[idx]), y[idx]
                half = torch.randint(0, N_CLASSES, (bs // 2,))
                yf = torch.cat([half, half])              # pairs of fakes with the same label
                z = torch.randn(bs, G.z_dim)
                fake = G(z, yf)
                # (a) discriminator, hinge loss: real scores above +1, fake scores below -1
                loss_d = torch.relu(1 - D(diff_augment(real), yr)).mean() +                     torch.relu(1 + D(diff_augment(fake.detach()), yf)).mean()
                optD.zero_grad(set_to_none=True)
                loss_d.backward()
                optD.step()
                # (b) generator: raise D's score on fakes, plus the mode-seeking term
                # (Mao et al., 2019): same label, different z -> images should differ
                adv = -D(diff_augment(fake), yf).mean()
                h = bs // 2
                ms = (fake[:h] - fake[h:]).abs().mean() / (z[:h] - z[h:]).abs().mean()
                loss_g = adv + args.ms_weight / (ms + 1e-5)
            else:
                real, yr = dihedral(normalise(x[idx])), y[idx]
                z = torch.randn(bs, G.z_dim)
                yf = torch.randint(0, N_CLASSES, (bs,))
                fake = G(z, yf)
                # real -> 0.9 (one-sided label smoothing), fake -> 0
                loss_d = bce(D(real, yr), torch.full((bs,), 0.9)) + bce(D(fake.detach(), yf), torch.zeros(bs))
                optD.zero_grad(set_to_none=True)
                loss_d.backward()
                optD.step()
                # non-saturating loss: make D say "real" (1) on fakes
                loss_g = bce(D(fake, yf), torch.ones(bs))
                ms = torch.zeros(())
            optG.zero_grad(set_to_none=True)
            loss_g.backward()
            optG.step()
            dsum, gsum, msum, nb = dsum + loss_d.item(), gsum + loss_g.item(), msum + ms.item(), nb + 1

        div = {CLASSES[c]: round(diversity(G, c), 2) for c in (0, 5)}
        history.append({"epoch": ep, "loss_d": dsum / nb, "loss_g": gsum / nb, "mode_seeking": msum / nb,
                        "diversity": div})
        print(f"ep {ep:3d} D {dsum / nb:.3f} G {gsum / nb:.3f} ms {msum / nb:.3f} diversity {div} "
              f"{time.time() - t0:.0f}s", flush=True)
        if ep % args.sample_every == 0 or ep == args.epochs:
            save_grid(G, f"results/cgan_samples_ep{ep:03d}.png")
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "ngf": args.ngf, "arch": args.arch,
                        "epoch": ep, "history": history}, "checkpoints/cgan.pt")
    return G


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ms_weight", type=float, default=0.2, help="weight of the mode-seeking term (sngan)")
    p.add_argument("--arch", default="sngan", choices=["sngan", "dcgan"],
                   help="sngan: spectral-norm projection discriminator + hinge loss (default); "
                        "dcgan: the first version, which mode-collapsed")
    p.add_argument("--sample_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    train(args)
