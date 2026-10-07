# BDH Narrative Consistency Classifier

An end-to-end Machine Learning & NLP pipeline designed to detect, analyze, and classify narrative consistency within text sequences using advanced transformer models and custom neural network architectures.

---

## 📌 Overview

Narrative consistency classification is crucial for evaluating long-form generated text, story plots, dialogue systems, and document verification. The **BDH Narrative Consistency Classifier** provides a structured framework to ingest text datasets, preprocess narrative pairs or sequences, and train classification models to distinguish between consistent and inconsistent narrative progressions.

---

## 📁 Repository Structure

```text
BDH-Narrative-Consistency-Classifier/
│
├── data/
│   └── README.md              # Instructions for dataset preparation & formatting
│
├── notebooks/
│   └── main_experiment.ipynb  # Interactive notebook for exploratory data analysis & experiments
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── config.py              # Global configurations, hyperparameters & paths
│   ├── dataset.py             # PyTorch Dataset & DataLoader utilities
│   ├── models.py              # Neural network architectures & transformer classifier models
│   ├── train.py               # Training, validation loop, and evaluation pipeline
│   └── utils.py               # Helper functions (logging, seed setup, metrics)
│
├── .gitignore                 # Files & directories excluded from version control
├── LICENSE                    # Repository license
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

---

## ✨ Features

- **Custom Dataset Pipeline:** Efficient tokenization and batch loading for narrative sentence pairs.
- **Modular Model Architectures:** Support for fine-tuning pretrained Transformers (BERT, RoBERTa, DeBERTa) and custom classification heads.
- **Config-Driven Experiments:** Centralized hyperparameter and path management in `src/config.py`.
- **Comprehensive Evaluation:** Accuracy, F1-Score, Precision, Recall tracking, and confusion matrix visualization.
- **Reproducibility:** Seed management integrated into model training pipelines.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Hrutik0555/BDH-Narrative-Consistency-Classifier.git
cd BDH-Narrative-Consistency-Classifier
```

### 2. Set Up a Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start & Usage

### 1. Data Preparation
Place your dataset files in the `data/` directory. Refer to [`data/README.md`](./data/README.md) for required schema details (e.g., `premise`, `hypothesis` / `narrative_1`, `narrative_2`, and `label`).

### 2. Configuration
Modify `src/config.py` to set target parameters such as model backbone, batch size, learning rate, and path settings:
```python
# Example config adjustments
MODEL_NAME = "roberta-base"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
```

### 3. Train the Model
Run the training module from the root directory:
```bash
python -m src.train
```

### 4. Interactive Experiments
Open Jupyter Notebook to explore data analysis, embedding visualizations, and preliminary model testing:
```bash
jupyter notebook notebooks/main_experiment.ipynb
```

---

## 📊 Model & Workflow Architecture

1. **Input Generation:** Narrative segments ($S_1, S_2$) are ingested and formatted into contextual pairs.
2. **Tokenization:** Inputs are tokenized using transformer-specific tokenizers with dynamic padding.
3. **Encoding:** Text sequences pass through fine-tuned transformer layers to extract contextualized representation vectors.
4. **Classification Head:** Linear layers with Dropout and GELU/ReLU activations predict consistency confidence scores.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve model efficiency, add new dataset loaders, or optimize training loops:
1. Fork the project repository.
2. Create your feature branch (`git checkout -b feature/NewFeature`).
3. Commit your changes (`git commit -m 'Add NewFeature'`).
4. Push to the branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
