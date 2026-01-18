# Nsight Compute CUDAGraph Kernel Breakdown

This guide captures per-kernel timing for CUDAGraph replay and plots time share at 256k.

## 1) Capture an NCU report (CUDAGraph replay)

Run a single 256k replay with minimal warmup:

```bash
ncu --target-processes all --force-overwrite --set full --replay-mode kernel \
  --section LaunchStats \
  --csv --page raw \
  -o ncu_attn_q2fp8_base_mask \
  python run_attn_bench_q2fp8_cudagraph.py \
    --attn-kernel attn_q2fp8_base_mask \
    --max-length 262144 \
    --step 262144 \
    --iters 1 \
    --warmup 0 \
    --cg-warmup 1 \
    --cg-replay-only \
    --no-plot \
    --no-flash \
  > ncu_attn_q2fp8_base_mask.csv
```

Notes:
- If your NCU build does not recognize `LaunchStats`, try `--section "Launch Statistics"` or remove `--section` and export from the UI instead.
- Use `--kernel-regex 'triton__|attn_'` in the `ncu` command if you want to limit to Triton kernels only.

UI alternative:
1. Run `ncu --target-processes all --force-overwrite -o ncu_attn_q2fp8_base_mask python run_attn_bench_q2fp8_cudagraph.py ...`.
2. Open `ncu_attn_q2fp8_base_mask.ncu-rep` in `ncu-ui`.
3. In the "Launch Statistics" page, export to CSV (use that CSV in the script below).

## 2) Plot time breakdown

```bash
python ncu_export_plot.py \
  --inputs ncu_attn_q2fp8_base_mask.csv ncu_attn_q2fp8_q2new.csv \
  --out ncu_kernel_breakdown_256k.png \
  --out-json ncu_kernel_breakdown_256k.json \
  --label-regex 'attn_q2fp8_[^._]+' \
  --sort asc
```

If your CSV does not contain units, pass `--unit us` (or `ns`/`ms`/`s`) explicitly.

## 3) Custom grouping (optional)

The script groups by kernel name substrings: threshold, stage1, stage2, scan, refine, and everything else goes to `other`.
You can override this with a JSON mapping:

```json
{
  "threshold": ["threshold"],
  "stage1": ["stage1"],
  "stage2": ["stage2"],
  "scan": ["scan"],
  "refine": ["refine"],
  "other": []
}
```

Then run:

```bash
python ncu_export_plot.py --inputs ... --group-map ncu_groups.json --out ncu_kernel_breakdown_256k.png
```
