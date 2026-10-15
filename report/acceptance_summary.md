# Day 10 acceptance status

## Training suite error

RuntimeError: Gradient overflow tại optimizer step 1; optimizer không cập nhật. Giảm LR hoặc kiểm tra số học trước khi chạy lại.

| Mã | Split | Params trainable | VRAM GB | Phút | Valid JSON | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| E0 | test |  |  |  | 0.900 | 0.100 | [0.073, 0.131] | 0.884 | 0.176 | 0.028 | 0.192 |

## Inputs and figures for REPORT.md

- Dataset: `d10-21e4070301`; session disjoint: True.
- train: 1422 ảnh, 3515 boxes, 13 sessions; boxes theo lớp: {'crack': 1750, 'manhole': 691, 'pothole': 1074}.
- val: 276 ảnh, 639 boxes, 2 sessions; boxes theo lớp: {'crack': 470, 'manhole': 103, 'pothole': 66}.
- test: 311 ảnh, 583 boxes, 3 sessions; boxes theo lớp: {'crack': 299, 'manhole': 163, 'pothole': 121}.
- E0 raw examples: `report/evidence/e0_raw_examples.json` (có).
- Coordinate round-trip: `report/coordinate_roundtrip.json` (có).
- Coordinate-scale mismatch diagnostic: `report/coordinate_scale_diagnostic.json` (có).
- Training example token/mask check: `report/training_example_check.json` (có).
- E1 architecture params: LLM+lm_head=4022468096, visual=415347712, projector=109105152; LoRA rank=16, target modules=252, projector params dtype=frozen, forward compute dtype=frozen.
- E1 loss: 1.4767 → 0.5622 (300 steps); log `work/e1_loss.json`.
- E2 architecture params: LLM+lm_head=4022468096, visual=415347712, projector=109105152; LoRA rank=16, target modules=252, projector params dtype=float32, forward compute dtype=float32.
- E2 loss: 1.3735 → 0.6918 (300 steps); log `work/e2_loss.json`.
- Figures: `report/figures/coord_roundtrip.png`, `report/figures/train_gt_examples.png`, `report/figures/training_loss.png`.

REPORT.md numeric table updated: False

## Rubric numeric gates

- BLOCKED: E1 parse_valid_rate >= 0.95
- BLOCKED: E1 F1 >= E0 F1 + 0.10
- FAIL: at least 6 annotated prediction images

## E5 diagnostic

- E5 reward-hacking audit: chưa có đủ evidence/prediction/test split.

## Download

After a full run, download `/content/K4-Track-4-Day10-VisionLanguageModel/report/day10_report_handoff.zip` from Colab Files or `MyDrive/AI20K_Lab_Day10_Checkpoints/day10_report_handoff.zip` from Drive; extract at repo root. The ZIP contains lab instructions and report artifacts, and excludes dataset, credentials and model weights.

## Official checker

```text
[PASS] evidence e0
[FAIL] evidence e1 — thiếu report/evidence/e1.json
[FAIL] evidence e2 — thiếu report/evidence/e2.json
[FAIL] evidence e3 — thiếu report/evidence/e3.json
[FAIL] evidence e4 — cần ít nhất một file e4*.json
[PASS] dataset_version nhất quán — ['d10-21e4070301']
[PASS] khớp data/splits/meta.json
[FAIL] REPORT.md đã điền — còn chuỗi '<điền'
[PASS] git commits — 19/5
[PASS] git forbidden files

5 FAIL, 0 WARN
```
