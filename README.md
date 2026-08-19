# fMRI Foundation Models for Single-Trial Visual Decoding

Research code for evaluating pretrained fMRI foundation models on
single-trial visual decoding with the Natural Scenes Dataset (NSD). The
repository studies two complementary questions within one experimental
framework:

1. whether a JEPA-style voxel encoder can transfer from pretraining to NSD
   beta maps and raw/segmented BOLD inputs; and
2. whether graph-structured brain representations can preserve
   trial-specific visual information for image retrieval.

The main downstream task is 100-way image retrieval against CLIP image
embeddings. The archive contains the final script retained for each reported
experiment; superseded debugging and intermediate variants are omitted.

## Repository structure

```text
.
├── SLIM_JEPA/
│   ├── preprocess/       # NSD trial preparation and CLIP targets
│   ├── input_variants/   # beta-repeat, raw-time, and segmented-beta inputs
│   ├── finetuning/       # frozen probe, LoRA, and full adaptation
│   ├── scaling/          # training-set scaling experiments
│   ├── evaluation/       # multi-seed and diagnostic evaluation
│   └── model/hiera/      # SLIM-compatible Hiera components
└── BrainGFM_Graph/
    ├── kmeans_beta/      # trial beta nodes with a fixed correlation graph
    ├── atlas_beta/       # AAL/Schaefer atlas controls
    ├── short_window_fc/  # short-window functional-connectivity control
    ├── category_probe/   # image-embedding cluster classification probe
    └── model/            # BrainGFM-compatible graph components
```

## Data and checkpoints

No participant data, COCO images, derived NSD arrays, pretrained weights, or
training checkpoints are included. Obtain NSD through its official access
procedure and comply with the applicable NSD and COCO terms.

The scripts expect the derived files named in their module docstrings, such as
single-trial beta maps, trial/image indices, CLIP embeddings, and official
train/test indices. Generated `.npy`, `.hdf5`, NIfTI, checkpoint, and log files
are excluded by `.gitignore`.

## Installation

Create separate Conda environments for the two routes:

```bash
conda env create -f SLIM_JEPA/environment.yml
conda env create -f BrainGFM_Graph/environment.yml
```

The exported environments record the software used on the original server.
CUDA/PyTorch builds may need to be adjusted for the local GPU driver.

## Path configuration

All machine-specific paths have been removed. Copy `.env.example` as a local
reference and export only the variables needed by the selected experiment:

| Variable | Purpose |
|---|---|
| `SLIM_NSD_DATA` | SLIM/JEPA derived NSD data directory |
| `SLIM_NSD_MIN_DATA` | beta HDF5 and `nsdgeneral.nii.gz` directory |
| `SLIM_NSD_RAW_BOLD` | raw subject-01 BOLD directory |
| `SLIM_BRAIN_CHECKPOINT` | pretrained SLIM checkpoint |
| `BRAINGFM_NSD_DATA` | BrainGFM experiment data directory |
| `BRAINGFM_PROCESSED_DATA` | processed graph/CLIP arrays |
| `NSD_MASK_PATH` | NSD general mask used by atlas conversion |
| `NSD_MNI_TO_FUNC_WARP` | official MNI-to-functional-space warp |
| `BRAINGFM_ATLAS_CACHE` | local AAL/Schaefer atlas cache |

If variables are unset, each route uses its own repository-local `data/`
directory and, for SLIM, `SLIM_JEPA/checkpoints/best_model.pth`.

Example:

```bash
export SLIM_NSD_DATA=/data/slim_nsd
export SLIM_NSD_MIN_DATA=/data/slim_nsd/nsd_subj01_min
export SLIM_BRAIN_CHECKPOINT=/models/slim/best_model.pth

python SLIM_JEPA/input_variants/extract_features.py
python SLIM_JEPA/scaling/probe_scaling_law_final.py --prepare-only
```

For the principal graph representation:

```bash
export BRAINGFM_NSD_DATA=/data/BrainGFM
export BRAINGFM_PROCESSED_DATA=/data/BrainGFM/processed

python BrainGFM_Graph/kmeans_beta/build_dataset.py
python BrainGFM_Graph/kmeans_beta/train_retrieval.py
python BrainGFM_Graph/kmeans_beta/eval_100way.py
```

Each script documents its required inputs and generated outputs at the top of
the file. Training and evaluation are GPU-oriented and were not rerun during
repository packaging because the restricted data and checkpoints are not part
of this archive.

## Experimental scope

The SLIM/JEPA route includes the final implementations used to compare input
construction, parameter adaptation, and data scaling. The graph route includes
the retained k-means beta representation and atlas, short-window FC, and
category-probe controls. Experimental outputs regarded as preliminary or
uncertain are not packaged as claims in this repository.

## Citation

GitHub can export the repository citation from `CITATION.cff`. When referring
to a fixed code snapshot, cite the versioned release/tag rather than the moving
`main` branch.

## Third-party code and terms

This research archive builds on
[SLIM](https://github.com/OneMore1/SLIM-Brain2026),
[BrainGFM](https://github.com/weixinxu666/BrainGFM),
[NSD](https://naturalscenesdataset.org/), and
[OpenCLIP](https://github.com/mlfoundations/open_clip). See
`THIRD_PARTY_NOTICE.md` for attribution and licensing notes. This repository
does not redistribute upstream datasets or pretrained weights and does not
purport to relicense third-party components.
