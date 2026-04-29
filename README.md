# Machine Learning Projects

Machine Learning projects developed during my Master's in NLP. Each project applies a different ML approach to text data, using real datasets and standard libraries from the Python ecosystem.

## Projects

| # | Project | Topic | Dataset |
|---|---------|-------|---------|
| 1 | [Statistical Model Comparison](./01_statistical_comparison/) | Paired statistical comparison of two classifiers (Logistic Regression vs. Gaussian Naive Bayes) | Spambase (OpenML) |
| 2 | [Multi-label Classification](./02_multilabel_classification/) | One-vs-Rest multi-label text classification | Jigsaw Toxic Comments |
| 3 | [Semi-Supervised Learning](./03_semisupervised_learning/) | Comparison of Self-Training and Clustering + Labeling against a supervised baseline | SMS Spam Collection |
| 4 | [Anomaly Detection](./04_anomaly_detection/) | Outlier detection on text data using Isolation Forest and Local Outlier Factor | 20 Newsgroups |

## Tech Stack

- **Language:** Python 3
- **Core libraries:** scikit-learn, pandas, NumPy, SciPy
- **NLP:** NLTK (tokenization, POS tagging, WordNet lemmatization)
- **Visualization:** matplotlib
- **Environment:** Google Colab

## How to Run

Each project folder contains its own README with specific instructions and a Jupyter notebook (`.ipynb`).

To run any notebook locally:

```bash
git clone https://github.com/Numidia2002/ML-projects.git
cd ML-projects/<project_folder>
jupyter notebook
```

Alternatively, open the notebooks directly in [Google Colab](https://colab.research.google.com/) by uploading the `.ipynb` file.

## About

These projects were developed as part of my Master's coursework in NLP. They reflect different machine learning paradigms (supervised, semi-supervised, unsupervised) applied to natural language data, with a focus on building complete pipelines: preprocessing, feature extraction, model training, evaluation, and statistical analysis.
