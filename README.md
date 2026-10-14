# ENG 2440 – Assignment 2: Tumour Segmentation in MRI Volumes

Student: Yomna Elnikety (25011181) – report: `YomnaElnikety_25011181_ENG2440_A2.pdf`.

## Repository contents

| File / folder | Purpose |
|---|---|
| `Assignment2_25011181.ipynb` | Fully executed notebook, Tasks A–G (inspection, split, preprocessing, training, sampling comparison, test evaluation, error analysis) |
| `a2_lib.py` | Supporting code, patch `Dataset` and samplers, 3D U-Net, loss, sliding-window inference, evaluation, training loop |
| `assignment2_split.csv` | Case-level split supplied by the course, unchanged|
| `environment.yml` | Software environment |
| `figures/` | Figures used in the report |
| `runs/`, `runs_p64/` | Training logs (`history.csv`, `train_log.csv`, `summary.json`), best checkpoints (`best.pt`) and test result tables for the $96^3$ and $64^3$ experiments |
| `ai_use_declaration.txt` | Declaration of AI assistance |

**Not included:** the MRI data, extracted patches, the preprocessed cache (`cache_preproc/`, about 15–20 GB)
and the `last.pt` checkpoints. See `.gitignore`.

## Setup

1. Create the environment (a GPU build of PyTorch is needed for training):
   ```bash
   conda env create -f environment.yml
   conda activate a2
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   The environment used for the reported results: Python 3.11.17, PyTorch 2.11.0+cu128, NumPy 2.4.6, nibabel 5.4.2.
2. Download `Task01_BrainTumour.tar` (about 7.6 GB)
   (`https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar`) and extract it:
   ```bash
   tar -xf Task01_BrainTumour.tar
   ```
   Only the 484 labelled cases in `imagesTr/` and `labelsTr/` are used, the unlabelled `imagesTs/` is never touched.
   Follow the dataset licence and attribution terms given in `Task01_BrainTumour/dataset.json`.
3. Put `assignment2_split.csv` and `a2_lib.py` in the same folder as the notebook.

## Exact run order

1. Open `Assignment2_25011181.ipynb` and change only `DATA_ROOT` in the cell marked
   *"Change only this line when running notebook"* (path to the folder containing `imagesTr/`, `labelsTr/`, `dataset.json`).
2. Run the notebook. It executes in this order:

   | Section | What it does | Approx. time |
   |---|---|---|
   | Setup | imports, seeds, environment log, load split table | seconds |
   | Task A | dataset.json, geometry of 3 cases, modality and three-plane figures | seconds |
   | Task B | split integrity checks, case counts, tumour-volume distribution | about 15 s |
   | Task C | geometry check of all cases, interpolation demo, normalisation, crop, cache of all 484 cases, augmentation figure | about 25 min (once, the cache is reused afterwards) |
   | Task D | dataset/batch check, model summary, sampler statistics, training of the uniform and tumour-aware runs | about 5.4 min per $96^3$ run (finished runs are skipped) |
   | Task E | validation comparison, $64^3$ sampler comparison (`runs_p64/`), final configuration | a few minutes per $64^3$ run |
   | Task F | sliding-window inference (validation, then test), postprocessing decision, test metrics, volume errors | about 25 s per split |
   | Task G | three-plane error figure, metric comparison, component and boundary analysis, duplicate-label check | about 1 min |

   The first cell of the training section runs a 20-iteration smoke test (`runs/smoke`) before the real runs.
   If `runs/` and `cache_preproc/` already exist, the notebook reuses them so delete these folders to retrain from scratch.

## Reproducibility details

- **Seed:** 42 for Python, NumPy and PyTorch, the `DataLoader` generator and per-worker generators are derived from it.
  cuDNN is set to deterministic mode, so repeated runs are very close but GPU arithmetic may still differ in the last digits.
- **Controlled comparison:** both samplers start from identical initial weights (hash `6cf75eb4726b`, checked in the notebook).
- **Hardware:** university lab workstation (`ws-l1-002`), NVIDIA RTX 5000 Ada (32 GB), 12 allocated CPU cores.
- **Training settings:** 3D U-Net (4 levels, 16-32-64-128 channels, 1,445,681 parameters), loss BCE + soft Dice,
  AdamW (peak lr 1e-3, weight decay 1e-5, cosine decay to 1%), 5,000 iterations, batch size 2, $96^3$ patches,
  fp16 mixed precision, left-right flip + intensity scale/shift + light noise on training patches only.
- **Inference:** $96^3$ windows, stride 48 (50% overlap), probabilities averaged before the 0.5 threshold,
  predictions pasted back onto the original $240\times240\times155$ grid.

## Headline results (test set, 73 cases, after postprocessing)

| Metric | Mean | SD | Median |
|---|---|---|---|
| Dice | 0.895 | 0.064 | 0.904 |
| IoU | 0.816 | 0.096 | 0.825 |
| Sensitivity | 0.903 | 0.083 | 0.929 |
| HD95 (mm) | 7.49 | 8.58 | 4.69 |

Absolute volume error: 11.70 mL on average (12.4%). The final configuration (uniform sampler, $96^3$ patches) was
chosen on validation results only, the test split was used once, in Task F. Full tables, figures and the error
analysis are in the report.

## AI use

See `AI_USE_DECLARATION.md`.
