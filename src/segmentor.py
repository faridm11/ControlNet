"""
Frozen segmentor used as the mask-adherence judge (mIoU) and for the Phase-2
segmentation-consistency loss.

Model: UNet++ / EfficientNet-B4, "sensation" checkpoint, 10 output classes,
trained with ImageNet normalisation. Its library (segmentation-models-pytorch)
is not part of the main environment; it is imported from SEG_DEPS_DIR.

Target mapping (dataset V1 class id -> segmentor class id) was derived from the
segmentor's own predictions on 300 real training images
(controlnet_testing/step2_segmentor, 2026-10-10):
  - a dataset class is assigned to a segmentor class when that class covers a
    clear majority of its pixels;
  - classes the segmentor confidently calls background (>= 90 %) are background;
  - everything ambiguous is IGNORED (neither rewarded nor penalised).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

SEG_DEPS_DIR = os.environ.get("DDPM_SEG_DEPS", "/home/woody/iwi5/iwi5388h/software/private/segdeps_smp034")
SEG_CHECKPOINT = os.environ.get(
    "DDPM_SEG_CHECKPOINT",
    str(Path(__file__).resolve().parents[2]
        / "sensation-models/UnetPlusPlus/efficientnet-b4/ckpt/sensation/epoch=251-val_loss=0.26557-val_iou=0.58698.ckpt"),
)

# Active judge, chosen per job with DDPM_SEG_JUDGE:
#   "sensation" (default) — the lab's 10-class UNet++ (weak on SANPO: mIoU 0.44 on real photos)
#   "mapillary"           — Mask2Former Swin-L trained on Mapillary Vistas (65 street-level classes),
#                           never trained on either of our datasets; works on both (see MAPILLARY_* below)
#   "ade20k"              — SegFormer-B5 trained on ADE20K (150 classes): the SECOND, independent judge.
#                           Different data and architecture than the Mapillary judge that drives the seg loss;
#                           never trained on our datasets. Same shared classes, so the same targets.
#   "cityscapes"          — SegFormer-B5 trained on Cityscapes (19 classes). Tested as second judge and rejected:
#                           car-view model, calls 83 % of SANPO sidewalk "road" (step 18).
SEG_JUDGE = os.environ.get("DDPM_SEG_JUDGE", "sensation")
SEG_IGNORE = 255
SEGFORMER_JUDGES = ("cityscapes", "ade20k")
SHARED_JUDGE_CLASSES = SEG_JUDGE in ("mapillary",) + SEGFORMER_JUDGES

if SHARED_JUDGE_CLASSES:
    # Shared judge classes for BOTH datasets; the last class "other" collects Mapillary classes we have no
    # counterpart for (snow, bench, trash can, ...): it never occurs in a target, so predicting it is wrong.
    # Crosswalks are scored as road: Mapillary labels only the painted stripes as crosswalk, our masks the
    # whole crossing area (first CPU test: crosswalk IoU 0.00 on both datasets for that reason alone).
    SEG_CLASS_NAMES = ["road", "sidewalk", "building", "fence/wall", "person", "vehicle",
                       "bicycle", "pole", "sign/light", "vegetation", "terrain", "sky", "other"]
else:
    SEG_CLASS_NAMES = ["background", "road", "sidewalk", "bikelane", "person", "car",
                       "bicycle", "traffic sign", "traffic light", "obstacle"]
SEG_NUM_CLASSES = len(SEG_CLASS_NAMES)

MAPILLARY_MODEL_ID = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
# Mapillary Vistas class id (65) -> shared judge class; anything not listed -> 13 "other"
_MV = {0: [13, 14, 10, 7, 12, 24, 41, 43, 36, 8, 23],  # road (+ crosswalk plain/markings, service lane, parking, bike lane, rail track, lane marking, manhole, pothole, catch basin)
       1: [15, 11, 2, 9],                       # sidewalk: sidewalk, pedestrian area, curb, curb cut
       2: [17, 18],                             # building, tunnel
       3: [3, 4, 5, 6],                         # fence, guard rail, barrier, wall
       4: [19, 20, 21, 22],                     # person, bicyclist, motorcyclist, other rider
       5: [55, 54, 61, 56, 60, 59, 58, 57],     # car, bus, truck, caravan, trailer, other vehicle, on rails, motorcycle
       6: [52],                                 # bicycle
       7: [45, 47, 44, 46],                     # pole, utility pole, street light, sign frame
       8: [50, 49, 48],                         # traffic sign front/back, traffic light
       9: [30],                                 # vegetation
       10: [29, 26],                            # terrain, sand
       11: [27]}                                # sky
MAPILLARY_TO_JUDGE = np.full(65, 12, dtype=np.int64)  # 12 = "other"
for _k, _ids in _MV.items():
    MAPILLARY_TO_JUDGE[_ids] = _k
# Our labels -> shared judge classes (255 = not scored: void, stairs, bridge, obstacle, ... no counterpart)
V1_TO_MAPILLARY_JUDGE = np.full(35, SEG_IGNORE, dtype=np.int64)
for _ids, _k in [((1, 3, 4, 5, 6), 0), ((2, 7), 1), ((18, 19, 23), 2), ((20, 21), 3), ((8, 9), 4),
                 ((10, 11, 12, 13, 14, 16, 17), 5), ((15,), 6), ((25,), 7), ((26, 27), 8), ((28,), 9), ((29,), 10), ((30,), 11)]:
    V1_TO_MAPILLARY_JUDGE[list(_ids)] = _k
SANPO_TO_MAPILLARY_JUDGE = np.full(31, SEG_IGNORE, dtype=np.int64)
for _ids, _k in [((1, 5, 19), 0), ((2, 3, 6, 17), 1), ((7, 10), 2), ((4, 8, 9, 11), 3), ((12, 13), 4),
                 ((21,), 5), ((24,), 7), ((22, 23), 8), ((28, 29), 9), ((30,), 10), ((27,), 11)]:
    SANPO_TO_MAPILLARY_JUDGE[list(_ids)] = _k

CITYSCAPES_MODEL_ID = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
# Cityscapes train id (19) -> shared judge class (no Cityscapes class maps to "other")
CITYSCAPES_TO_JUDGE = np.array([0, 1, 2, 3, 3, 7, 8, 8, 9, 10, 11, 4, 4, 5, 5, 5, 5, 5, 6], dtype=np.int64)
#  road sidewalk building wall fence pole light sign vegetation terrain sky person rider car truck bus train motorcycle bicycle

ADE20K_MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"
# ADE20K class id (150, 0-based as in the model's id2label) -> shared judge class; anything not listed -> 12 "other"
_ADE = {0: [6],                                  # road
        1: [11, 52],                             # sidewalk, path
        2: [1, 25, 48, 14, 8, 79, 84, 86],       # building, house, skyscraper, door, windowpane, hovel, tower, awning
        3: [0, 32, 38, 95],                      # wall, fence, railing, bannister
        4: [12],                                 # person
        5: [20, 80, 83, 102, 116],               # car, bus, truck, van, minibike
        6: [127],                                # bicycle
        7: [93, 87],                             # pole, streetlight
        8: [43, 136],                            # signboard, traffic light
        9: [4, 17, 72, 66],                      # tree, plant, palm, flower
        10: [9, 13, 29, 46, 94, 91],             # grass, earth, field, sand, land, dirt track
        11: [2]}                                 # sky
ADE20K_TO_JUDGE = np.full(150, 12, dtype=np.int64)
for _k, _ids in _ADE.items():
    ADE20K_TO_JUDGE[_ids] = _k
# Mapping version 2 (step 19, DDPM_JUDGE_MAPPING=v2): raw classes that name the right thing but fell into "other"
# (diagnosed on real SANPO photos). v1 stays the default so earlier numbers remain reproducible.
JUDGE_MAPPING = os.environ.get("DDPM_JUDGE_MAPPING", "v1")
MAPILLARY_TO_JUDGE_V2 = MAPILLARY_TO_JUDGE.copy()
MAPILLARY_TO_JUDGE_V2[28] = 10                               # Snow -> terrain (snow-covered lawns)
ADE20K_TO_JUDGE_V2 = ADE20K_TO_JUDGE.copy()
ADE20K_TO_JUDGE_V2[[3, 140]] = 1                             # floor, pier -> sidewalk (paved plazas, promenades)
ADE20K_TO_JUDGE_V2[42] = 7                                   # column -> pole
if JUDGE_MAPPING == "v2":
    MAPILLARY_TO_JUDGE, ADE20K_TO_JUDGE = MAPILLARY_TO_JUDGE_V2, ADE20K_TO_JUDGE_V2
_SEGFORMER = {"cityscapes": (CITYSCAPES_MODEL_ID, CITYSCAPES_TO_JUDGE), "ade20k": (ADE20K_MODEL_ID, ADE20K_TO_JUDGE)}

_ASSIGNED = {1: 1, 2: 2, 4: 3, 10: 5, 17: 5, 15: 6, 26: 7, 25: 9}   # road, sidewalk, cycling lane, car, trailer, bicycle, sign, pole
_BACKGROUND = {11, 12, 18, 19, 20, 22, 23, 24, 28, 29, 30}           # truck, bus, building, door, wall, bridge, tunnel, stairs, vegetation, earth, sky
# Ignored: unlabeled, crosswalk, parking_driveway, railtrack, curb, person, rider, tramtrain,
# motorbike, caravan, fence, traffic_light and the four void classes.

V1_TO_SEG = np.full(35, SEG_IGNORE, dtype=np.int64)
for _c, _k in _ASSIGNED.items():
    V1_TO_SEG[_c] = _k
for _c in _BACKGROUND:
    V1_TO_SEG[_c] = 0

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class MapillaryJudge(torch.nn.Module):
    """Mask2Former (Mapillary Vistas) as a per-pixel classifier over the shared judge classes.

    Input: ImageNet-normalised images (what segmentor_logits passes). The image is upscaled by
    DDPM_SEG_INPUT_SCALE (default 2: 512 -> 1024 px, closer to Mapillary's training resolution).
    Output: log-probabilities over the 13 judge classes (12 + "other") at input resolution. Semantic probabilities
    follow Mask2Former's own semantic inference (softmax over classes x sigmoid masks); Mapillary
    classes are summed into judge classes. Differentiable, so it could also drive a Phase-2 loss.
    """

    def __init__(self, model, scale):
        super().__init__()
        self.model, self.scale = model, scale
        self.register_buffer("to_judge", torch.nn.functional.one_hot(
            torch.from_numpy(MAPILLARY_TO_JUDGE), SEG_NUM_CLASSES).float())  # (65, 14)

    def forward(self, x):
        h, w = x.shape[-2:]
        if self.scale != 1:
            x = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        out = self.model(pixel_values=x)
        cls = out.class_queries_logits.softmax(-1)[..., :-1]          # (B, Q, 65) without "no object"
        masks = out.masks_queries_logits.sigmoid()                     # (B, Q, h/4, w/4)
        sem = torch.einsum("bqc,bqhw->bchw", cls, masks)              # (B, 65, h/4, w/4)
        sem = torch.nn.functional.interpolate(sem, size=(h, w), mode="bilinear", align_corners=False)
        judge = torch.einsum("bchw,ck->bkhw", sem, self.to_judge)     # (B, 13, h, w)
        judge = judge / judge.sum(1, keepdim=True).clamp_min(1e-8)
        return judge.clamp_min(1e-8).log()


class SegformerJudge(torch.nn.Module):
    """SegFormer (Cityscapes or ADE20K) as a per-pixel classifier over the shared judge classes.

    Input: ImageNet-normalised images (SegFormer's own normalisation), upscaled by DDPM_SEG_INPUT_SCALE.
    Output: log-probabilities over the 13 judge classes at input resolution (model softmax summed into
    judge classes via `to_judge`).
    """

    def __init__(self, model, scale, to_judge):
        super().__init__()
        self.model, self.scale = model, scale
        self.register_buffer("to_judge", torch.nn.functional.one_hot(
            torch.from_numpy(to_judge), SEG_NUM_CLASSES).float())  # (n model classes, 13)

    def forward(self, x):
        h, w = x.shape[-2:]
        if self.scale != 1:
            x = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        logits = self.model(pixel_values=x).logits                     # (B, 19, H/4, W/4)
        logits = torch.nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        judge = torch.einsum("bchw,ck->bkhw", logits.softmax(1), self.to_judge)
        return judge.clamp_min(1e-8).log()


def segmentor_available() -> bool:
    if SHARED_JUDGE_CLASSES:
        return True  # loaded from the Hugging Face cache
    return Path(SEG_DEPS_DIR).is_dir() and Path(SEG_CHECKPOINT).is_file()


def load_segmentor(device):
    """Build the segmentor, load the checkpoint, freeze it and put it in eval mode."""
    if SEG_JUDGE == "mapillary":
        from transformers import Mask2FormerForUniversalSegmentation
        m2f = Mask2FormerForUniversalSegmentation.from_pretrained(MAPILLARY_MODEL_ID)
        model = MapillaryJudge(m2f, float(os.environ.get("DDPM_SEG_INPUT_SCALE", "2")))
        model.requires_grad_(False)
        return model.eval().to(device)
    if SEG_JUDGE in SEGFORMER_JUDGES:
        from transformers import SegformerForSemanticSegmentation
        model_id, to_judge = _SEGFORMER[SEG_JUDGE]
        seg = SegformerForSemanticSegmentation.from_pretrained(model_id)
        model = SegformerJudge(seg, float(os.environ.get("DDPM_SEG_INPUT_SCALE", "2")), to_judge)
        model.requires_grad_(False)
        return model.eval().to(device)

    if SEG_DEPS_DIR not in sys.path:
        sys.path.append(SEG_DEPS_DIR)
    import segmentation_models_pytorch as smp

    model = smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights=None,
                             in_channels=3, classes=SEG_NUM_CLASSES)
    state = torch.load(SEG_CHECKPOINT, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[len("whole_model."):]: v for k, v in state.items() if k.startswith("whole_model.")})
    model.requires_grad_(False)
    return model.eval().to(device)


def segmentor_logits(segmentor, images_01: torch.Tensor) -> torch.Tensor:
    """Run the segmentor on images in [0, 1], shape (B, 3, H, W). Gradients flow to the input."""
    mean = images_01.new_tensor(_MEAN).view(1, 3, 1, 1)
    std = images_01.new_tensor(_STD).view(1, 3, 1, 1)
    return segmentor((images_01 - mean) / std)


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(SEG_NUM_CLASSES x SEG_NUM_CLASSES) counts, rows = target, ignoring SEG_IGNORE pixels."""
    valid = target != SEG_IGNORE
    idx = target[valid] * SEG_NUM_CLASSES + pred[valid]
    return torch.bincount(idx, minlength=SEG_NUM_CLASSES ** 2).view(SEG_NUM_CLASSES, SEG_NUM_CLASSES)


def miou_from_confusion(conf: torch.Tensor):
    """Mean IoU over the classes that occur in the target. Returns (miou, pixel_acc, per-class dict)."""
    conf = conf.double()
    tp = conf.diag()
    union = conf.sum(0) + conf.sum(1) - tp
    present = conf.sum(1) > 0
    ious = {SEG_CLASS_NAMES[k]: (tp[k] / union[k]).item() for k in range(SEG_NUM_CLASSES) if present[k]}
    miou = float(np.mean(list(ious.values()))) if ious else 0.0
    acc = (tp.sum() / conf.sum()).item() if conf.sum() > 0 else 0.0
    return miou, acc, ious
