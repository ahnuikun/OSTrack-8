# B1 + MSTF Full-Training Manual

This project is the server-adapted OSTrack checkout. Do not change the existing
dataset paths, local environment files, or server launch assumptions when adding
or testing MSTF.

## Purpose

Run one full OSTrack-style training to compare the fast-screen MSTF conclusion
against real full-data training.

- Baseline B1 config: `vitb_256_mae_ce_32x4_ep300_fulltn`
- Candidate config: `vitb_256_mae_ce_32x4_ep300_fulltn_mstf`
- Only architecture change: enable MSTF before the OSTrack box head
- Training data and schedule remain the B1/full OSTrack setting:
  `LASOT + GOT10K_vottrain + COCO17 + TRACKINGNET`, ratio `1:1:1:1`,
  `60000` samples per epoch, `300` epochs

## Training

Use the existing server-adapted training entry. Example for 4 GPUs:

```bash
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300_fulltn_mstf --save_dir ./output --mode multiple --nproc_per_node 4 --use_lmdb 0 --use_wandb 0
```

Expected checkpoint directory:

```text
output/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_ep300_fulltn_mstf/
```

## Testing Gate

Always report detailed numbers. Do not summarize with only "better" or "worse".
For every dataset, report at least:

```text
Dataset | AUC | baseline AUC | delta AUC | Precision | baseline Precision | delta Precision | Norm Precision | baseline Norm Precision | delta Norm Precision
```

Required test order:

1. Test and analyze `visdrone`, `uavdt`, and `dtb70` first.
2. Report all three dataset results with exact metric values and deltas.
3. If both `visdrone` and `uavdt` improve by at least `+0.5 AUC`, do not make
   `dtb70` a hard blocker, but still keep its result in the report.
4. Only then test `uav123`.
5. If `uav123` also improves by at least `+0.5 AUC`, test `lasot`.
6. Final report must include one combined table for every tested dataset and a
   clear gate decision.

Example commands:

```bash
python tracking/test_uav_suite.py --tracker_param vitb_256_mae_ce_32x4_ep300_fulltn_mstf --dataset visdrone --num_gpus 4
python tracking/test_uav_suite.py --tracker_param vitb_256_mae_ce_32x4_ep300_fulltn_mstf --dataset uavdt --num_gpus 4
python tracking/test_uav_suite.py --tracker_param vitb_256_mae_ce_32x4_ep300_fulltn_mstf --dataset dtb70 --num_gpus 4

python tracking/analyze_uav_suite.py --tracker_param vitb_256_mae_ce_32x4_ep300_fulltn_mstf --dataset visdrone --force_evaluation
python tracking/analyze_uav_suite.py --tracker_param vitb_256_mae_ce_32x4_ep300_fulltn_mstf --dataset uavdt --force_evaluation
python tracking/analyze_uav_suite.py --tracker_param vitb_256_mae_ce_32x4_ep300_fulltn_mstf --dataset dtb70 --force_evaluation
```

Conditional follow-up:

```bash
python tracking/test_uav_suite.py --tracker_param vitb_256_mae_ce_32x4_ep300_fulltn_mstf --dataset uav123 --num_gpus 4
python tracking/analyze_uav_suite.py --tracker_param vitb_256_mae_ce_32x4_ep300_fulltn_mstf --dataset uav123 --force_evaluation

python tracking/test_uav_suite.py --tracker_param vitb_256_mae_ce_32x4_ep300_fulltn_mstf --dataset lasot --num_gpus 4
python tracking/analyze_uav_suite.py --tracker_param vitb_256_mae_ce_32x4_ep300_fulltn_mstf --dataset lasot --force_evaluation
```

## Baseline Reference

Use the corresponding B1/full-training baseline result from
`vitb_256_mae_ce_32x4_ep300_fulltn` when computing deltas. If a dataset is
re-evaluated, use the newly computed baseline table and keep the exact date in
the experiment notes.
