# Known issue: mask-after-inversion in step 3 (fixed 2026-06-14)

## Summary
`03_channel_splitting` step 3 inverted the membrane channel **before** the cell mask was
re-applied. The pipeline computes, effectively:

```
membrane_for_LocThk = max_val - (membrane_prediction × cell_mask)
```

Because the area outside the nephrocyte was set to 0 by the cell-mask multiplication, the
subsequent global inversion (`max_val - x`) flips that exterior 0 to **max_val (white)**. Local
Thickness measures *white* structures, so the entire outside-the-cell region is then measured as a
single enormous spurious "slit gap".

## Symptom
On a representative cell the outside-cell region was 100% white in the Local Thickness input and
produced thickness values with **mean ~528 px, max ~747 px** (vs. genuine inside-cell slit widths of
~15–40 px). In the Local Thickness output, ~44% of all non-zero pixels came from this exterior sea.

## Why it biases results (and isn't fully cancelled by the 0–100 px binning)
Step 5 histograms non-zero pixels into 0–100 px bins, so most of the exterior sea (>100 px) is
discarded. **But:**
1. A thin rim of the exterior is <100 px and leaks into the histogram; and
2. Where the segmented membrane does not fully seal the cell perimeter, the exterior white sea
   floods perimeter slit gaps that lie **inside** the cell mask, inflating genuine measurements near
   the boundary.

The net effect is an **upward bias of ~2–5 px in per-cell mean SD width, and it is non-uniform**
(larger for lines with wider gaps / leakier perimeters), so it distorts between-condition
comparisons rather than adding a constant offset.

## Fix
Re-apply the cell mask **after** inversion (implemented in `03_channel_splitting.ipynb`, cell 4):

```python
channel_1 = max_val - channel_1
cell_mask = tifffile.imread(padding_file) > 0          # {base_name}_padding.tif in 002_padding
channel_1 = np.where(cell_mask, channel_1, 0).astype(img.dtype)
```

After the fix the exterior is exactly 0 (background, ignored by Local Thickness), so the giant
spurious gap and the perimeter flooding both disappear.

## Verification (Lukas RNAi screen dataset)
- Outside-cell white pixels: **100% → 0%**; max Local Thickness **747 → 67 px**.
- Fraction of non-zero pixels >100 px: **0.44 → 0.00**.
- Inside-cell pixels: **bit-for-bit identical** to before (real signal untouched).
- Per-line means dropped ~2–5 px with the **ranking preserved** (e.g. Control 20.9 → 18.1 px;
  strongest wideners Rab11-2 / Rab11-1 / Anx-B10-II unchanged).

## ⚠️ Impact on already-published results
Any dataset processed with the pre-fix pipeline — **including the published sister dataset** — carries
this upward bias. Reproducing the corrected values requires re-running **steps 3 → 4 → 5 → 6**
(step 4 is the manual Fiji Local Thickness step). A correction/erratum to reported absolute SD widths
may be warranted; qualitative conclusions and rank order are expected to hold.

A pre-fix copy of the notebook is kept alongside as
`03_channel_splitting.ipynb.bak_pre_maskfix_2026-06-14`.
