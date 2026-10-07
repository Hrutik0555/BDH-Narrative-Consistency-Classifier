# BDH-Narrative-Consistency-Classifier

A PyTorch pipeline that decides whether a piece of narrative text is **consistent** or **contradicts** a given caption/claim, by tracking a dynamic *belief state* sentence by sentence and measuring the "friction" each new sentence causes.

## Overview

1. A frozen `sentence-transformers/all-MiniLM-L6-v2` encoder embeds the caption (initial belief state) and every sentence of the narrative (events).
2. The **BDH core** (a gated recurrent update) revises the belief state after each sentence. The L2 norm of each update is the **friction**.
3. Per-sample friction statistics (mean, variance) are concatenated with the final belief state and fed to a small classifier head.
4. Predictions are written with a short rationale based on a friction threshold calibrated on validation data.

## Training pipeline

| Phase | Description | Optimised parameters |
|-------|-------------|----------------------|
| 1. Unsupervised pretraining | Minimise friction loss on all training narratives (labels unused) | BDH core |
| 2. Supervised fine-tuning | Cross-entropy + `0.2 x` friction auxiliary loss, cosine LR schedule, best checkpoint by validation accuracy | BDH core + classifier |

## Project structure

```
BDH-Narrative-Consistency-Classifier/
├── src/
│   ├── config.py      # hyperparameters
│   ├── utils.py       # sentence splitting, seeding, label helper
│   ├── models.py      # BDHCore, BDHModel
│   ├── dataset.py     # NarrativeDataset
│   └── train.py       # training + evaluation + inference entrypoint
├── data/              # put train.csv and test.csv here
├── notebooks/main_experiment.ipynb
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/<your-username>/BDH-Narrative-Consistency-Classifier.git
cd BDH-Narrative-Consistency-Classifier
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Add `train.csv` and `test.csv` to `data/` (see [`data/README.md`](data/README.md)).
2. Run from the repository root:

```bash
python -m src.train
```

Output: `RES1.csv` with columns `id`, `label` (0 = consistent, 1 = contradiction), `rationale`.

Hyperparameters live in [`src/config.py`](src/config.py).

## Notes and limitations

- The encoder is frozen; only the BDH core and classifier head are trained.
- The original experiment used a very small dataset (~80 training and 8 validation samples), so the reported validation accuracy (0.75) is **not statistically meaningful**.
- The pretraining objective only minimises friction, which on its own can push updates towards zero; treat it as a regulariser and compare against `PRETRAIN_EPOCHS = 0`.
- Sentences are embedded one at a time, so long narratives are slow on CPU.

## License

Released under the [MIT License](LICENSE).
