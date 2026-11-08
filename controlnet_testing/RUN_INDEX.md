# Run index

Generated 2026-11-08 15:29 by `reporting/index_runs.py`. One row per training run; the run id is in every file name of that run (`logs/metrics_<run>.csv`, `logs/<run>_config.json`, `logs/<run>_final_eval.json`, `plots/<run>_*.png`, `samples/<run or job…>_epoch…/`, `checkpoints/<run>_LoRA.pt`). Runs on dataset 2 start with `sanpo_`.

Merged model copies (`<run>_Full/`, 1.4 GB each) of superseded runs were deleted on 2026-10-17 to save space; the LoRA file `<run>_LoRA.pt` is kept and rebuilds them (`src/models/diffusion_controlnet.py: load_checkpoint` + `create_pipeline`).

| Run | Dataset | Name | Folder | Settings | Epochs | Best KID | Final eval | Checkpoint files kept | Notes |
|---|---|---|---|---|---|---|---|---|---|
| sanpo_1843536 | sanpo | final_sanpo | `controlnet_testing/FINAL/sanpo/run` | LR 0.0001, r16, prompts_meta | 8 | 0.0093 @ ep 8 | KID 0.0093, mIoU 0.397 | sanpo_1843536_Full, sanpo_1843536_LoRA.pt |  |
| 1843532 | sidewalk | final_sidewalk | `controlnet_testing/FINAL/sidewalk/run` | LR 0.0001, r16, prompts_objects | 20 | 0.0129 @ ep 20 | KID 0.0129, mIoU 0.446 | 1843532_Full, 1843532_LoRA.pt |  |
| sanpo_1848221 | sanpo | v2_sanpo | `controlnet_testing/FINAL_v2/sanpo/run` | LR 0.0001, r16, prompts_meta | 8 | 0.0093 @ ep 8 | KID 0.0093, mIoU 0.505 | sanpo_1848221_Full, sanpo_1848221_LoRA.pt |  |
| 1848218 | sidewalk | v2_sidewalk | `controlnet_testing/FINAL_v2/sidewalk/run` | LR 0.0001, r16, prompts_objects | 20 | 0.0178 @ ep 12 | KID 0.0178, mIoU 0.567 | 1848218_Full, 1848218_LoRA.pt |  |
| sanpo_1848504 | sanpo | v2_segloss_sanpo | `controlnet_testing/FINAL_v2_segloss/sanpo/run` | LR 0.0001, r16, prompts_meta, seg-loss 0.0→0.006 | 8 | 0.0096 @ ep 8 | KID 0.0096, mIoU 0.551 | sanpo_1848504_Full, sanpo_1848504_LoRA.pt |  |
| sanpo_1850042 | sanpo | v2_segloss_l002_sanpo | `controlnet_testing/FINAL_v2_segloss/sanpo_l0.02/run` | LR 0.0001, r16, prompts_meta, seg-loss 0.0→0.02 | 8 | 0.0135 @ ep 8 | KID 0.0135, mIoU 0.578 | sanpo_1850042_Full, sanpo_1850042_LoRA.pt, sanpo_1850042_ep2_LoRA.pt, sanpo_1850042_ep4_LoRA.pt, sanpo_1850042_ep6_LoRA.pt, sanpo_1850042_ep8_LoRA.pt |  |
| sanpo_1858162 | sanpo | v2_segloss_l0025_0045_sanpo | `controlnet_testing/FINAL_v2_segloss/sanpo_l0.025-0.045/run` | LR 0.0001, r16, prompts_meta, seg-loss 0.025→0.045, checkpoint = last | 8 | 0.0215 @ ep 2 | KID 0.0251, mIoU 0.597 | sanpo_1858162_Full, sanpo_1858162_LoRA.pt, sanpo_1858162_ep2_LoRA.pt, sanpo_1858162_ep4_LoRA.pt, sanpo_1858162_ep6_LoRA.pt, sanpo_1858162_ep8_LoRA.pt |  |
| sanpo_1848904 | sanpo | v2_segloss_ramp_sanpo | `controlnet_testing/FINAL_v2_segloss/sanpo_ramp/run` | LR 0.0001, r16, prompts_meta, seg-loss 0.05→0.4, checkpoint = last | 8 | 0.0248 @ ep 2 | KID 0.0418, mIoU 0.719 | sanpo_1848904_Full, sanpo_1848904_LoRA.pt, sanpo_1848904_ep2_LoRA.pt, sanpo_1848904_ep4_LoRA.pt, sanpo_1848904_ep6_LoRA.pt, sanpo_1848904_ep8_LoRA.pt |  |
| 1843542 | sidewalk | screen_attn_ff | `controlnet_testing/step11_hparam_screen/attn_ff` | LR 0.0001, r16, prompts_objects | 10 | 0.0139 @ ep 10 | KID 0.0139, mIoU 0.452 | 1843542_LoRA.pt |  |
| 1843543 | sidewalk | screen_hflip | `controlnet_testing/step11_hparam_screen/hflip` | LR 0.0001, r16, prompts_objects | 10 | 0.0185 @ ep 10 | KID 0.0185, mIoU 0.424 | 1843543_LoRA.pt |  |
| 1843541 | sidewalk | screen_rank32 | `controlnet_testing/step11_hparam_screen/rank32` | LR 0.0001, r32, prompts_objects | 10 | 0.0168 @ ep 10 | KID 0.0168, mIoU 0.415 | 1843541_LoRA.pt |  |
| 1843540 | sidewalk | screen_reference | `controlnet_testing/step11_hparam_screen/reference` | LR 0.0001, r16, prompts_objects | 10 | 0.0182 @ ep 10 | KID 0.0182, mIoU 0.434 | 1843540_LoRA.pt |  |
| sanpo_1859598 | sanpo | final_gen200_sanpo | `controlnet_testing/step20_downstream_sanpo/generator_200/run` | LR 0.0001, r16, prompts_meta, seg-loss 0.025→0.045, checkpoint = last | 15 | 0.0295 @ ep 5 | KID 0.0301, mIoU 0.594 | sanpo_1859598_Full, sanpo_1859598_LoRA.pt, sanpo_1859598_ep10_LoRA.pt, sanpo_1859598_ep15_LoRA.pt, sanpo_1859598_ep5_LoRA.pt |  |

## Evaluation result folders (three-seed and standalone evaluations)

| Folder | JSON files |
|---|---|
| `controlnet_testing/FINAL/sanpo/eval_mapillary` | 6 |
| `controlnet_testing/FINAL/sidewalk/eval_mapillary` | 6 |
| `controlnet_testing/FINAL_v2/sanpo/eval` | 3 |
| `controlnet_testing/FINAL_v2/sidewalk/eval` | 3 |
| `controlnet_testing/FINAL_v2_segloss/sanpo/eval` | 3 |
| `controlnet_testing/FINAL_v2_segloss/sanpo_l0.02/eval` | 3 |
| `controlnet_testing/FINAL_v2_segloss/sanpo_l0.025-0.045/eval` | 3 |
| `controlnet_testing/FINAL_v2_segloss/sanpo_ramp/eval` | 3 |
| `controlnet_testing/step18_second_judge/eval` | 6 |

## Downstream segmentor results (`step20_downstream_sanpo/results/`, 522 real SANPO test photos)

| Setting | Start weights | Real photos | Test mIoU per seed |
|---|---|---|---|
| real_only | ade20k | limited_real | 0.623 / 0.624 / 0.616 |
| real_only_200 | ade20k | limited_200 | 0.627 / 0.622 / 0.617 |
| real_plus_base | ade20k | limited_real | 0.614 / 0.606 / 0.610 |
| real_plus_base_200 | ade20k | limited_200 | 0.605 / 0.594 / 0.590 |
| real_plus_ours | ade20k | limited_real | 0.630 / 0.624 / 0.626 |
| real_plus_ours_200 | ade20k | limited_200 | 0.614 / 0.601 / 0.597 |
