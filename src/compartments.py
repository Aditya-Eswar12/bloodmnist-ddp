"""
Element A: stain-derived compartment maps.

Splits a Giemsa-stained blood-cell image into three compartments with no labels
and no learned model:

    0 = background (plasma, red blood cells, anything that is not the cell)
    1 = cytoplasm
    2 = nucleus

The idea is Beer-Lambert: the more stain a pixel holds, the less light it lets
through. We convert RGB to optical density (absorbance) relative to the empty
background, then use two facts about the stain:

  * the azure / methylene-blue component binds DNA and absorbs RED light, so the
    nucleus is the region with the highest red-channel absorbance;
  * white-cell cytoplasm also takes up some azure, while red blood cells are
    full of haemoglobin (which absorbs violet-blue light strongly) and take up
    eosin (which absorbs green-blue light). So a stained pixel that absorbs more
    red than blue light is cytoplasm, and one that absorbs more blue than red is
    a red cell. (On the dev images, red-light and blue-light absorbance each
    separated cytoplasm from red cells with AUC above 0.93.)

`patch_fractions` then turns the pixel map into per-patch fractions, which is
what the masking policy consumes.
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_multiotsu

BACKGROUND, CYTOPLASM, NUCLEUS = 0, 1, 2

STAIN_MIN = 0.15      # total absorbance below this = unstained plasma
RIM_FACTOR = 2.0      # cell may extend this many nucleus-radii beyond the nucleus


def optical_density(img):
    """RGB uint8 (H, W, 3) -> per-channel absorbance (H, W, 3), float32.

    The background colour is taken as the 95th percentile of each channel,
    i.e. the brightest thing that is reliably present: unstained plasma.
    """
    x = img.astype(np.float32) + 1.0
    bg = np.percentile(x.reshape(-1, 3), 95, axis=0)
    return -np.log(np.minimum(x / bg, 1.0))


def _keep_large(mask, min_size):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.where(sizes >= min_size)[0])


def segment(img):
    """RGB uint8 (H, W, 3) -> label map (H, W) with values {0, 1, 2}."""
    h, w = img.shape[:2]
    empty = np.zeros((h, w), np.uint8)
    od = gaussian(optical_density(img), sigma=h / 128.0, channel_axis=-1)
    density = od.sum(-1)                          # total stain per pixel
    azure = od[..., 0]                            # red-light absorbance
    purple = od[..., 0] - od[..., 2]              # red minus blue absorbance

    # nucleus: top class of a 3-level Otsu threshold on red-light absorbance
    try:
        _, t_hi = threshold_multiotsu(azure, classes=3)
    except ValueError:                            # flat image, nothing to split
        return empty
    nucleus = ndi.binary_opening(azure > t_hi)
    nucleus = _keep_large(nucleus, max(3, 0.004 * h * w))
    nucleus = ndi.binary_fill_holes(nucleus)
    if not nucleus.any():
        return empty

    # cell body: stained, more azure than eosin, attached to the nucleus,
    # and not further out than a generous cytoplasm rim
    dist = ndi.distance_transform_edt(~nucleus)
    radius = np.sqrt(nucleus.sum() / np.pi)
    cell = (density > STAIN_MIN) & (purple > 0.0)
    cell &= dist <= RIM_FACTOR * radius + 0.03 * h
    cell = ndi.binary_opening(cell) | nucleus
    lab, _ = ndi.label(cell)
    keep = np.unique(lab[nucleus])
    cell = ndi.binary_fill_holes(np.isin(lab, keep[keep > 0]))

    out = empty
    out[cell] = CYTOPLASM
    out[nucleus] = NUCLEUS
    return out


def patch_fractions(label_map, patch=8):
    """Label map (H, W) -> (N, 3) array of [background, cytoplasm, nucleus]
    fractions per patch, patches in row-major order (same order as the ViT)."""
    h, w = label_map.shape
    gh, gw = h // patch, w // patch
    blocks = label_map[:gh * patch, :gw * patch].reshape(gh, patch, gw, patch)
    fr = np.stack([(blocks == k).mean(axis=(1, 3)) for k in (0, 1, 2)], axis=-1)
    return fr.reshape(gh * gw, 3).astype(np.float32)
