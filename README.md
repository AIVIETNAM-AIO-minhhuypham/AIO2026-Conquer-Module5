# ViT vs ResNet on CIFAR-10 with limited data

How much more does a small ViT lose than a CNN of the same size when the training set shrinks, and how much of that gap does strong augmentation close?

- Models (trained from scratch, ~2.7M params each): ViT with 4x4 patches, ViT with 8x8 patches, ResNet-18 at half width with a CIFAR stem.
- Data: 45k train pool / 5k validation (stratified, fixed) / official 10k test. Nested 10% ⊂ 30% ⊂ 100% subsets drawn per seed.
- Augmentation: `basic` (crop + flip) vs `strong` (basic + RandAugment + Mixup/CutMix).
- Fixed number of updates (30k steps, batch 128) at every data size.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
# install torch + torchvision with CUDA from https://pytorch.org/get-started/locally
pip install -r requirements.txt
```

## Usage

```bash
python main.py sanity                        # split checks, param counts, overfit a 256-image batch
python main.py overfit --overfit-steps 1500
python main.py bench                         # throughput + estimated GPU hours per plan
python main.py queue --plan lr_sweep         # 3 lrs x 2 models, validation only
python main.py queue --plan lr_sweep_ext
python main.py select-lr                     # writes result/chosen_lr.json
python main.py queue --plan pilot10          # 10% data, seed 0, validation only
python main.py queue --plan tier1            # main runs: 2 models x {10%,100%} x {basic,strong} x 3 seeds
python main.py queue --plan tier2            # ViT patch 8
python main.py queue --plan tier3            # 30% data
python main.py queue --plan budget_check     # 100% data with 2x steps
python main.py summarize                     # result/summary.csv
```

Finished runs are skipped and interrupted runs resume from `last.pt`, so the same command can simply be re-run.

Figures are not produced during training. Once results exist, use the functions in `vitcnn/plots.py` (see `notebooks/analysis.ipynb`).

Everything is written to `result/`: `runs/<run_id>/` (log.csv, metrics.json, best.pt, test predictions), `sanity/`, `bench/`, `splits/`, `figures/`, `summary.csv`, `console.log`.

## Status

- [x] data splits, models, training loop, checks, throughput benchmark
- [x] lr search, pilot at 10% data
- [ ] extend ResNet lr grid (4e-3), re-run overfit check with more steps
- [ ] tier1 / tier2 / tier3 main runs
- [ ] aggregation over seeds, McNemar, attention distance
