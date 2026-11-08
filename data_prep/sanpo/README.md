# Dataset 2: SANPO-Real (chest camera)

Preparation of SANPO-Real (Google Research, CC BY 4.0, https://google-research-datasets.github.io/sanpo_dataset/)
as the second dataset for this project. The processed data has the same folder layout as dataset 1 (`DDPM/data/`),
but its labels keep SANPO's own 31 class ids, and each class is conditioned with an ADE20K colour
(`class_mapping_sanpo.py`). `data_sanpo/LABEL_SPACE` tells `src/config.py` to use these definitions, so every
script switches datasets with one environment variable:

    export DDPM_DATA_ROOT=/home/woody/iwi5/iwi5388h/SENSATION/DDPM/data_sanpo

## Folder layout

| Path | What |
|---|---|
| `DDPM/data/` | dataset 1, segments/sidewalk-semantic (unchanged) |
| `DDPM/data_sanpo/{train,test}/images/` | dataset 2 photos, 1920×1080 PNG |
| `DDPM/data_sanpo/{train,test}/labels/` | dataset 2 masks in SANPO's own 31 class ids, 1920×1080 PNG |
| `DDPM/data_sanpo/LABEL_SPACE` | `sanpo` — selects the SANPO class definitions in `src/config.py` |
| `DDPM/data_sanpo/{train,test}/prompts/` | prompt CSVs (`prompts.csv`, `prompts_objects.csv`, `prompts_meta.csv`) |
| `DDPM/data_sanpo/metadata.csv` | per frame: split, session, frame number, weather, visibility, environment |
| `DDPM/data_prep/sanpo/` | the scripts below and `selection.csv` (which frames were used) |
| `/home/woody/iwi5/iwi5388h/datasets/sanpo_raw/` | SANPO label map, split lists, session index, original masks of the selected frames (outside the repo) |

## Steps

Run from `DDPM/` with `https_proxy=http://proxy.nhr.fau.de:80` (login node, no GPU needed):

1. `python data_prep/sanpo/index_sessions.py` — reads metadata of all 701 sessions; 116 train / 30 test sessions have chest-camera masks.
2. `python data_prep/sanpo/select_frames.py` — human-annotated chest frames only, thinned to every 3rd (train, ~1.2 s apart) and every 5th (test, ~2 s apart): 3,172 train / 522 test frames. Official SANPO session splits, so train and test never share a session.
3. `python data_prep/sanpo/download_and_convert.py` — downloads the selected frames, resizes to 1920×1080.
4. `python data_prep/sanpo/fetch_raw_masks.py` — downloads SANPO's original masks of those frames (class id per pixel).
5. `python data_prep/sanpo/build_labels.py` — writes the labels (SANPO ids, 1920×1080) and the `LABEL_SPACE` marker.
6. Prompts: `python data_prep/generate_prompts.py --data-root data_sanpo --label-space sanpo --crop 512 --min-area 0.002 --variant {full,objects,meta} --out-name <file>`.
7. Checks: `python data_prep/sanpo/verify_mapping_against_ade20k.py` (colours vs the official ADE20K table) and
   `DDPM_DATA_ROOT=$PWD/data_sanpo python data_prep/sanpo/check_dataset.py`.

## Class mapping (SANPO → ADE20K colours)

Defined in `class_mapping_sanpo.py`, checked by `verify_mapping_against_ade20k.py` against the official 150-class ADE20K
table (`data_prep/color_coding_semantic_segmentation_classes.xlsx`, exported to `data_prep/ade20k_150_palette.csv`).

- **Exact (20 classes, 87 % of pixels):** road, sidewalk, guard rail → railing, paved trail → path, building, wall/fence → fence,
  hand rail → bannister/handrail, door, pedestrian → person, animal, stairs, water body → water, vehicle → car, traffic sign → signboard,
  traffic light, pole, sky, tree, vegetation → plant, terrain → grass.
- **Closest (7 classes, 6 %):** curb → sidewalk, crosswalk → road, other walkable surface → sidewalk, opening-gate → fence,
  rider → person, railway track → road, bike rack → railing. ADE20K has no class for these.
- **Void, white (4 classes, 6 %):** unlabeled, inaccessible surface, bus stop, obstacle. "Obstacle" is mostly snow piles,
  plus bins and barriers; ADE20K has neither snow nor a generic obstacle class.

Choices for unclear classes were made after looking at examples (wall/fence is mostly metal fences, terrain mostly lawns,
vegetation mostly bushes and hedges). For prompts and the Phase-2 segmentor target the labels are converted to the
project's 35-class ids with `SANPO_TO_V1`. SANPO has no bicycle or bike-lane class.

## Weather

Sessions with chest-camera masks are labelled sunny, cloudy or snowy (about one third snowy); none are rainy or low-visibility.
