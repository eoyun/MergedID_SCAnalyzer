# Detector-Separated EB/EE Weighting, Classification, and Fusion

This directory contains two workflows:

1. The current workflow for building sample-balanced pT weights, training EB/EE image and track classifiers, and combining them with a stacking ensemble.
2. An older mass-regression workflow that uses a different HDF5 format and hard-coded paths.

The current workflow is the one to use for the EB/EE separation work.

## Files

- `build_sample_pt_weights.py`
  Builds weights for `A_pT` entries only. Each sample gets total weight sum `1`, and each non-empty pT bin inside that sample gets equal weight sum.
- `build_event_level_weights.py`
  Builds event-level sidecar weights and separates the balancing into three cases:
  - overall event case
  - `EB` case
  - `EE` case
- `train_resnet_image_classifier.py`
  Trains a binary classifier on `EB` or `EE` detector images using the sidecar HDF5 produced by `build_event_level_weights.py`.
- `train_point_transformer_track_classifier.py`
  Trains a binary point-cloud transformer on `EB` or `EE` track points using the same sidecar HDF5.
- `train_fusion_ensemble.py`
  Loads one trained image run and one trained track run, re-scores the shared validation/test splits, then trains a stacking ensemble on the validation split.
- `track_point_transformer.py`
  Shared dataset, point-cloud construction, and model code used by the track trainer and the ensemble script.
- `torch_resnet_concat.py`
  Legacy custom ResNet used by the older regression scripts.
- `train_regression.py`
  Legacy regression trainer with hard-coded input paths.
- `test_regression.py`
  Legacy regression inference script with hard-coded model and test-file paths.

## Environment

The code in this directory was run with the local virtual environment:

```bash
./.venv/bin/python -V
```

The active scripts require:

- `python>=3.10`
- `h5py`
- `numpy`
- `matplotlib`
- `torch`
- `torchvision`
- `scikit-learn`

If `matplotlib` complains about a non-writable config directory, use:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
mkdir -p "$MPLCONFIGDIR"
```

In this workspace, prefer running scripts as:

```bash
./.venv/bin/python <script>.py ...
```

## Input Dataset Layout

The weighting scripts expect an input directory with this structure:

```text
<input-dir>/
  W/
    *.h5
  Z/
    *.h5
  signal/
    *.h5
```

Background is formed by combining all files under `W/` and `Z/`.

Signal sample IDs are parsed from the source ROOT path and become names like:

- `signal_H250_A0p4`
- `signal_H750_A10`
- `signal_H2000_A5`

The code expects 16 sample IDs in total:

- `background`
- 15 signal `(H, A)` combinations

## pT Bins

The weighting scripts use these pT bin edges:

```text
[0, 20, 30, 50, 60, 70, 80, 100, 125, 150, 200, 250, 300, 350, 400, 500, 600, 800, 1000, 1200, 1500, 2000, inf)
```

For every sample:

- the total weight sum is `1`
- every non-empty pT bin inside that sample gets the same total weight sum

For `build_event_level_weights.py`, that balancing is done separately for:

- `event`
- `EB`
- `EE`

## How EB/EE-Separated Weighting Works

The important point is that `EB` and `EE` are not weighted by reusing one shared detector-independent object weight.

Instead, the script solves three separate balancing problems:

1. `event` case
   Uses all selected events in the sample.
2. `EB` case
   Uses only selected events that contain at least one `EB` object.
3. `EE` case
   Uses only selected events that contain at least one `EE` object.

For each sample ID independently:

- `background`
- each of the 15 signal `(H, A)` samples

the script does the following for each case independently:

1. Derive one representative event pT from `A_pT` and `A_event_idx`.
2. Select the events belonging to that case.
3. Count how many selected events fall into each pT bin.
4. Give every non-empty pT bin the same total weight sum.
5. Force the total weight sum over that sample and case to be exactly `1`.

That means:

- `background` has total weight sum `1` in the `event` case.
- `background` also has total weight sum `1` in the `EB` case.
- `background` also has total weight sum `1` in the `EE` case.
- The same is true for every signal sample.

These are separate normalizations. They are not required to match event-by-event between `event`, `EB`, and `EE`.

### Event membership for the detector-separated cases

- If an event has at least one `EB` object, it contributes to the `EB` balancing.
- If an event has at least one `EE` object, it contributes to the `EE` balancing.
- If an event has both `EB` and `EE` objects, it contributes to both balancing problems independently.
- If an event has no `EB` objects, it does not contribute to the `EB` normalization.
- If an event has no `EE` objects, it does not contribute to the `EE` normalization.

### Meaning of the saved weights

- `event_weight`
  Per-event weight from the detector-independent `event` normalization.
- `EB_event_weight`
  Per-event weight from the `EB` normalization only.
- `EE_event_weight`
  Per-event weight from the `EE` normalization only.
- `A_weight_duplicate`
  Copies `event_weight` onto every `A` object in the event.
- `A_weight_split`
  Splits `event_weight` equally across all `A` objects in the event.
- `EB_weight_duplicate`
  Copies `EB_event_weight` onto every `EB` object in the event.
- `EB_weight_split`
  Splits `EB_event_weight` equally across all `EB` objects in the event.
- `EE_weight_duplicate`
  Copies `EE_event_weight` onto every `EE` object in the event.
- `EE_weight_split`
  Splits `EE_event_weight` equally across all `EE` objects in the event.

### Which weight should be used for training

- Train the `EB` classifier with `EB_weight_split`.
- Train the `EE` classifier with `EE_weight_split`.

This is the correct detector-separated choice because the total object weight inside each sample is inherited from the detector-specific event balancing, not from the global `event` balancing.

### What is equalized

Inside one fixed sample and one fixed case:

- the total weight sum is `1`
- every non-empty pT bin gets the same total weight sum

Example:

- if sample `signal_H750_A10` has 14 non-empty pT bins in the `EB` case, then each non-empty `EB` pT bin gets total weight sum `1/14`
- if the same sample has 15 non-empty pT bins in the `EE` case, then each non-empty `EE` pT bin gets total weight sum `1/15`

So `EB` and `EE` can legitimately have different per-bin target sums for the same sample, because they are normalized independently.

## Recommended Workflow

### 1. Build detector-separated event weights

This is the main sidecar used by the classifier.

Example:

```bash
./.venv/bin/python build_event_level_weights.py \
  --input-dir /home/eoyun/data/260616_v2_Mini \
  --output-dir outputs/test_event_background_1M_sep \
  --background-max-events 100000
```

What it does:

- reads all `W`, `Z`, and `signal` HDF5 files
- derives one representative event pT from `A_pT` and `A_event_idx`
- balances the total sample weight to `1`
- equalizes the total weight across non-empty pT bins
- repeats that balancing independently for:
  - all selected events
  - events with at least one `EB` object
  - events with at least one `EE` object

Important behavior:

- `event_weight` is the overall event-balanced weight.
- `EB_event_weight` is the event-balanced weight for the `EB` case only.
- `EE_event_weight` is the event-balanced weight for the `EE` case only.
- `EB_weight_split` divides `EB_event_weight` equally across all `EB` objects in the event.
- `EE_weight_split` divides `EE_event_weight` equally across all `EE` objects in the event.

For classifier training, use:

- `EB_weight_split` for `--detector eb`
- `EE_weight_split` for `--detector ee`

### 2. Train a real EB model

For real training, do not pass `--debug`.

The classifier supports these model backbones:

- `resnet18`
- `resnet34`
- `resnet50`

These are torchvision architectures initialized from scratch inside the script. They are not ImageNet-pretrained in the current code.

Recommended first full training command:

```bash
./.venv/bin/python train_resnet_image_classifier.py \
  --detector eb \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/train_resnet_eb \
  --model resnet18 \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --monitor auc \
  --patience 5
```

If you want a larger model:

```bash
./.venv/bin/python train_resnet_image_classifier.py \
  --detector eb \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/train_resnet_eb_r34 \
  --model resnet34 \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --monitor auc \
  --patience 5
```

Or:

```bash
./.venv/bin/python train_resnet_image_classifier.py \
  --detector eb \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/train_resnet_eb_r50 \
  --model resnet50 \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --monitor auc \
  --patience 5
```

### 3. Train a real EE model

Recommended first full training command:

```bash
./.venv/bin/python train_resnet_image_classifier.py \
  --detector ee \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/train_resnet_ee \
  --model resnet18 \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --monitor auc \
  --patience 5
```

If you want a larger model:

```bash
./.venv/bin/python train_resnet_image_classifier.py \
  --detector ee \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/train_resnet_ee_r34 \
  --model resnet34 \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --monitor auc \
  --patience 5
```

Or:

```bash
./.venv/bin/python train_resnet_image_classifier.py \
  --detector ee \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/train_resnet_ee_r50 \
  --model resnet50 \
  --epochs 30 \
  --batch-size 8 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --monitor auc \
  --patience 5
```

### 4. Optional smoke test

Use this only to confirm the pipeline works before a full training run.

EB:

```bash
./.venv/bin/python train_resnet_image_classifier.py \
  --detector eb \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/debug_resnet_eb_sep \
  --epochs 1 \
  --batch-size 2 \
  --debug \
  --debug-max-events-per-sample 12
```

EE:

```bash
./.venv/bin/python train_resnet_image_classifier.py \
  --detector ee \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/debug_resnet_ee_sep \
  --epochs 1 \
  --batch-size 2 \
  --debug \
  --debug-max-events-per-sample 12
```

## Output of `build_event_level_weights.py`

The script writes:

- `event_level_weights.h5`
- `sample_event_bin_summary.csv`
- `file_event_summary.csv`
- `config.json`

### `event_level_weights.h5`

Top-level structure:

```text
event_level_weights.h5
  sample_summary/
    event/
      <sample_id>/
    eb/
      <sample_id>/
    ee/
      <sample_id>/
  files/
    W/
      <stem>/
    Z/
      <stem>/
    signal/
      <stem>/
```

Inside `sample_summary/<case>/<sample_id>/`:

- `event_bin_counts`
- `event_bin_weight_sums`
- `event_weight_lookup`

Inside `files/<split>/<stem>/`:

- `event_pt`
- `event_pt_bin`
- `event_selected_mask`
- `event_weight`
- `A_weight_duplicate`
- `A_weight_split`
- `EB_event_selected_mask`
- `EB_event_weight`
- `EB_weight_duplicate`
- `EB_weight_split`
- `EE_event_selected_mask`
- `EE_event_weight`
- `EE_weight_duplicate`
- `EE_weight_split`

Key meaning:

- `duplicate` means the event weight is copied to every object in that collection.
- `split` means the event weight is divided equally among the objects in that collection for that event.

For the current classifier code, `split` is the correct choice.

### `sample_event_bin_summary.csv`

One row per:

- balancing case: `event`, `EB`, or `EE`
- sample
- pT bin

Columns:

- `case`
- `sample_id`
- `bin_index`
- `pt_low`
- `pt_high`
- `event_count`
- `event_bin_weight_sum`
- `per_event_weight`

### `file_event_summary.csv`

One row per input HDF5 file.

Columns include:

- `selected_events`
- `selected_eb_events`
- `selected_ee_events`
- `n_a`
- `n_eb`
- `n_ee`

### `config.json`

Stores:

- pT bin edges
- sample IDs
- selected background size
- available balanced cases
- default classifier weight keys

## Output of `train_resnet_image_classifier.py`

Each run writes to the directory passed by `--output-dir`.

Typical files:

- `best_model.pt`
- `run_config.json`
- `manifest_summary.json`
- `history.json`
- `training_history.png`
- `test_metrics.json`
- `test_predictions.csv`
- `roc_test.png`
- `score_distribution_test.png`
- `efficiency_vs_pt_test.json`
- `efficiency_vs_pt_test.png`

### What the classifier does

- reads the weight sidecar and raw HDF5 paths from `event_level_weights.h5`
- picks `EB_weight_split` or `EE_weight_split` by default
- builds event-level train/val/test splits per sample
- rescales object weights inside each split so `background` and `all signal combined` have the same total weight sum
- converts sparse detector data into dense `512 x 512` images
- trains a binary classifier with weighted BCE loss
- chooses the operating threshold from the validation set
- evaluates weighted AUC and weighted F1 on the test set

### Important training options

- `--detector`
  Must be `eb` or `ee`. Train them in separate runs.
- `--model`
  Chooses the classifier backbone: `resnet18`, `resnet34`, or `resnet50`.
- `--epochs`
  Maximum number of epochs. Training can stop earlier because of early stopping.
- `--batch-size`
  Increase this if GPU memory allows. For `resnet50`, a smaller batch size is usually safer.
- `--num-workers`
  Data loading workers. Start with `4` if multiprocessing is available. In restricted environments, the script now falls back automatically to `0`.
- `--monitor`
  Early stopping metric. `auc` is the current default and usually the better first choice.
- `--patience`
  Number of epochs without improvement before stopping.
- `--disable-log-scale`
  Turns off `log1p` preprocessing for detector channels. Leave this off unless you want an explicit comparison.
- `--threshold-mode`
  Threshold choice on validation set. Default is `max-f1`.
- `--weight-key`
  Override the weight dataset in the sidecar. Normally leave this unset so the script uses `EB_weight_split` or `EE_weight_split`.
- `--debug`
  Only for smoke tests. Do not use it for real training.

### How to choose a model

- Start with `resnet18` if you want the simplest stable baseline.
- Use `resnet34` if you want more capacity and training time is acceptable.
- Use `resnet50` only if your GPU memory is large enough and you want the heaviest backbone currently supported.

### Training behavior

- The script automatically uses `cuda` if available, otherwise it runs on CPU.
- If PyTorch multiprocessing is blocked by the environment, the script automatically falls back to `num_workers=0` instead of crashing.
- The script now caps raw HDF5 handles per dataset worker and uses PyTorch `file_system` sharing when available to reduce file-descriptor pressure.
- The script saves the best checkpoint to `best_model.pt` according to `--monitor`.
- The script does not currently support resume-from-checkpoint training. Each run starts from scratch in a new output directory.
- Train, validation, and test splits are created per sample, not globally across all objects.
- After each split manifest is built, the trainer renormalizes weights so the total `background` weight equals the total `signal` weight in that split while preserving relative weights inside each class.

### Detector channels

For `EB`, the image has 4 channels:

- `SC_energy`
- `EB_track_pt_Lost`
- `EB_track_pt_PF`
- `EB_track_pt_GSF`

For `EE`, the image has 6 channels:

- `EE_seed_energy`
- `ES_seed_plane1_energy`
- `ES_seed_plane2_energy`
- `EE_track_pt_Lost`
- `EE_track_pt_PF`
- `EE_track_pt_GSF`

Sparse track features are expanded into dense `512 x 512` maps.

## Track Point-Cloud Transformer

`train_point_transformer_track_classifier.py` is the track-only branch. It uses the same object manifests, the same event-level split logic, and the same detector-specific weights as the image trainer.

### Point-cloud definition

For each `EB` or `EE` object, the script reads:

- `*_track_pt_GSF_idx` and `*_track_pt_GSF_val`
- `*_track_pt_PF_idx` and `*_track_pt_PF_val`
- `*_track_pt_Lost_idx` and `*_track_pt_Lost_val`

where `*` is `EB` or `EE`.

Each sparse track entry becomes one point with 6 input features:

1. `x`
   Derived from `flat_idx % 512`, then normalized to `[-1, 1]`.
2. `y`
   Derived from `flat_idx // 512`, then normalized to `[-1, 1]`.
3. `track_pt_feature`
   By default `log1p(pT)`. Use `--disable-log-track-pt` to switch to raw `pT`.
4. `is_gsf`
5. `is_pf`
6. `is_lost`

So the track branch does not train three separate models for `GSF`, `PF`, and `Lost`. It trains one model on one merged point cloud and lets the one-hot type bits tell the model which track family each point belongs to.

### Track truncation

- The default `--max-points` is `16`.
- If an object has more than `--max-points` total track points, the script keeps the highest-`pT` points.
- Empty point clouds are allowed. In that case the transformer still sees its learned class token, so inference does not crash.

### Track model architecture

The track model is a small point-cloud transformer:

- linear projection from 6 input features to `--embed-dim`
- learned class token
- `--depth` transformer encoder blocks
- `--num-heads` attention heads per block
- MLP expansion controlled by `--mlp-ratio`
- final MLP head that outputs one logit

The default configuration is:

- `--embed-dim 128`
- `--depth 4`
- `--num-heads 4`
- `--mlp-ratio 4.0`
- `--dropout 0.1`

### What the track trainer writes

Each run directory contains:

- `best_model.pt`
- `run_config.json`
- `manifest_summary.json`
- `history.json`
- `training_history.png`
- `test_metrics.json`
- `val_predictions.csv`
- `test_predictions.csv`
- `roc_test.png`
- `score_distribution_test.png`
- `efficiency_vs_pt_test.json`
- `efficiency_vs_pt_test.png`

The track prediction CSVs include the usual object identifiers plus:

- `n_gsf`
- `n_pf`
- `n_lost`
- `sum_pt_gsf`
- `sum_pt_pf`
- `sum_pt_lost`

These extra fields are later reused by the fusion step.

### Train the track branch

Example full training:

```bash
./.venv/bin/python train_point_transformer_track_classifier.py \
  --detector eb \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/train_point_transformer_eb \
  --epochs 20 \
  --batch-size 64 \
  --num-workers 4

./.venv/bin/python train_point_transformer_track_classifier.py \
  --detector ee \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/train_point_transformer_ee \
  --epochs 20 \
  --batch-size 64 \
  --num-workers 4
```

Example smoke test:

```bash
./.venv/bin/python train_point_transformer_track_classifier.py \
  --detector eb \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/debug_point_transformer_eb_sep \
  --epochs 1 \
  --batch-size 4 \
  --debug \
  --debug-max-events-per-sample 12
```

### Important track options

- `--max-points`
  Maximum number of merged track points kept per object.
- `--disable-log-track-pt`
  Uses raw `pT` instead of `log1p(pT)`.
- `--embed-dim`, `--depth`, `--num-heads`, `--mlp-ratio`, `--dropout`
  Transformer architecture controls.
- `--monitor`
  Early stopping metric, same meaning as in the image trainer.
- `--threshold-mode`
  Validation rule used to choose the final operating point.

## Image + Track Stacking Ensemble

`train_fusion_ensemble.py` is the fusion step. It does not train the base models from scratch. Instead it takes:

- one completed image run directory from `train_resnet_image_classifier.py`
- one completed track run directory from `train_point_transformer_track_classifier.py`

and then:

1. verifies that both runs use the same detector, weight sidecar, split seed, split fractions, and debug settings
2. rebuilds the exact same validation and test object sets
3. loads both checkpoints
4. recomputes image scores and track scores on `val` and `test`
5. trains a weighted logistic-regression stacking model on the `val` split
6. applies the fitted fusion model to the `test` split

### Fusion features

The stacking model uses 8 features:

1. `image_logit`
2. `track_logit`
3. `log1p_n_gsf`
4. `log1p_n_pf`
5. `log1p_n_lost`
6. `log1p_sum_pt_gsf`
7. `log1p_sum_pt_pf`
8. `log1p_sum_pt_lost`

The script standardizes these features and fits a weighted `LogisticRegression` model from `scikit-learn`.

### Why the fusion is trained on validation, not test

The ensemble weights must be learned on a split that is separate from the final test set. That is why the script uses:

- `val` to fit the logistic-regression stacker and choose the final threshold
- `test` only for the final reported performance

This avoids leaking test information into the ensemble coefficients.

### Run the ensemble

Example:

```bash
./.venv/bin/python train_fusion_ensemble.py \
  --image-run-dir outputs/train_resnet_eb \
  --track-run-dir outputs/train_point_transformer_eb \
  --output-dir outputs/fusion_eb \
  --num-workers 4

./.venv/bin/python train_fusion_ensemble.py \
  --image-run-dir outputs/train_resnet_ee \
  --track-run-dir outputs/train_point_transformer_ee \
  --output-dir outputs/fusion_ee \
  --num-workers 4
```

Example smoke test:

```bash
./.venv/bin/python train_fusion_ensemble.py \
  --image-run-dir outputs/debug_resnet_eb_sep \
  --track-run-dir outputs/debug_point_transformer_eb_sep \
  --output-dir outputs/debug_fusion_eb_sep
```

### What the ensemble writes

- `run_config.json`
- `manifest_summary.json`
- `stacking_coefficients.json`
- `test_metrics.json`
- `val_predictions.csv`
- `test_predictions.csv`
- `roc_test.png`
- `score_distribution_test.png`
- `efficiency_vs_pt_test.json`
- `efficiency_vs_pt_test.png`

`stacking_coefficients.json` records:

- the feature names
- the fitted logistic-regression coefficients
- the intercept
- the mean and scale used by the `StandardScaler`

### Important ensemble requirement

The image and track runs must be compatible. In practice that means:

- same `--detector`
- same `--weight-h5`
- same `--weight-key`
- same `--seed`
- same `--train-frac`
- same `--val-frac`
- same `--debug`
- same `--debug-max-events-per-sample`

If these do not match, the script stops instead of silently combining mismatched models.

## Optional: Build A-level pT weights

`build_sample_pt_weights.py` is separate from the detector classifier workflow.

Use it if you want a sidecar for `A_pT` entries themselves.

Example:

```bash
./.venv/bin/python build_sample_pt_weights.py \
  --input-dir /home/eoyun/data/260616_v2_Mini \
  --output-dir outputs/test_background_1M \
  --background-max-entries 100000
```

This writes:

- `A_pT_weights.h5`
- `sample_bin_summary.csv`
- `file_summary.csv`
- `config.json`

This script does not create detector-separated `EB` and `EE` weights.

## Legacy Regression Workflow

The following files are older and are not part of the new detector-separated classifier pipeline:

- `train_regression.py`
- `test_regression.py`
- `torch_resnet_concat.py`

Important limitations:

- they use a different HDF5 schema
- they expect datasets like `all_jet`, `am`, `iphi`, and `ieta`
- they rely on hard-coded file paths
- they write to `MODELS/`, `LOGS/`, and `PLOTS/`
- they are GPU-oriented and assume `.cuda()` is available

Use them only if you are working with the old mass-regression dataset and are prepared to edit paths manually.

## Common Commands

Rebuild the detector-separated sidecar:

```bash
./.venv/bin/python build_event_level_weights.py \
  --input-dir /home/eoyun/data/260616_v2_Mini \
  --output-dir outputs/test_event_background_1M_sep \
  --background-max-events 100000
```

Run full training:

```bash
./.venv/bin/python train_resnet_image_classifier.py \
  --detector eb \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/train_resnet_eb \
  --model resnet18 \
  --epochs 20 --batch-size 16 --num-workers 4

./.venv/bin/python train_resnet_image_classifier.py \
  --detector ee \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/train_resnet_ee \
  --model resnet18 \
  --epochs 20 --batch-size 16 --num-workers 4

./.venv/bin/python train_point_transformer_track_classifier.py \
  --detector eb \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/train_point_transformer_eb \
  --epochs 20 --batch-size 64 --num-workers 4

./.venv/bin/python train_point_transformer_track_classifier.py \
  --detector ee \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/train_point_transformer_ee \
  --epochs 20 --batch-size 64 --num-workers 4

./.venv/bin/python train_fusion_ensemble.py \
  --image-run-dir outputs/train_resnet_eb \
  --track-run-dir outputs/train_point_transformer_eb \
  --output-dir outputs/fusion_eb \
  --num-workers 4

./.venv/bin/python train_fusion_ensemble.py \
  --image-run-dir outputs/train_resnet_ee \
  --track-run-dir outputs/train_point_transformer_ee \
  --output-dir outputs/fusion_ee \
  --num-workers 4
```

Run smoke tests:

```bash
./.venv/bin/python train_resnet_image_classifier.py \
  --detector eb \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/debug_resnet_eb_sep \
  --epochs 1 --batch-size 2 --debug --debug-max-events-per-sample 12

./.venv/bin/python train_resnet_image_classifier.py \
  --detector ee \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/debug_resnet_ee_sep \
  --epochs 1 --batch-size 2 --debug --debug-max-events-per-sample 12

./.venv/bin/python train_point_transformer_track_classifier.py \
  --detector eb \
  --weight-h5 outputs/test_event_background_1M_sep/event_level_weights.h5 \
  --output-dir outputs/debug_point_transformer_eb_sep \
  --epochs 1 --batch-size 4 --debug --debug-max-events-per-sample 12

./.venv/bin/python train_fusion_ensemble.py \
  --image-run-dir outputs/debug_resnet_eb_sep \
  --track-run-dir outputs/debug_point_transformer_eb_sep \
  --output-dir outputs/debug_fusion_eb_sep
```

## Troubleshooting

- If `python` is not found, use `./.venv/bin/python`.
- If plots fail with a Matplotlib config permission warning, set `MPLCONFIGDIR=/tmp/matplotlib`.
- If `num_workers > 0` is not allowed on your machine, the training script now falls back automatically to `num_workers=0`.
- If you hit `OSError: [Errno 24] Too many open files`, update to the latest script and, if needed, reduce `--num-workers` first.
- If training is slow, reduce `--batch-size`, reduce `--num-workers`, or start with `--debug`.
- If a sample has no entries in a detector case, the weight builder will raise an error because equal per-sample balancing is impossible for that case.
- If you want to override the default weight dataset in the classifier, use `--weight-key`.
- If the fusion script reports incompatible runs, retrain one branch so the image and track models share the same split settings.

## Verified Example Outputs in This Workspace

These outputs were generated successfully in this directory:

- `outputs/test_event_background_1M_sep/`
- `outputs/debug_resnet_eb_sep/`
- `outputs/debug_resnet_ee_sep/`
- `outputs/debug_point_transformer_eb_sep/`
- `outputs/debug_fusion_eb_sep/`

The detector-separated sidecar was verified so that, for every sample:

- total weight sum is `1.0` in the `event` case
- total weight sum is `1.0` in the `EB` case
- total weight sum is `1.0` in the `EE` case
- non-empty pT bins inside each case have equal total weight sum
