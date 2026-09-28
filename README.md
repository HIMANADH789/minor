# ECG_Benchmark

A PyTorch time-series benchmark workspace for comparing ECG5000 and real bearing-fault processing pipelines, including transport-style and baseline model variants.

## Required Python environment

This project is written for a Python 3.10+ environment and uses the following main packages:

```bash
pip install torch numpy scipy pandas scikit-learn matplotlib requests rich tqdm aeon
```

Recommended extras for model evaluation and experiment output:

```bash
pip install pytest seaborn jupyter ipykernel
```

Notes:

- The code imports `torch`, `numpy`, `scipy`, `requests`, `rich`, `aeon`, and `sklearn` directly.
- The repository’s dataset loaders and training scripts expect project-local data files and an active `numpy`/`PyTorch` environment.

## Exact datasets required

This repository is built around two benchmark datasets:

### 1) ECG5000 dataset (required time-series benchmark)

The primary benchmark dataset is the ECG5000 time-series classification dataset from the UCR/AEON time-series archive. The downloader script in this repository is:

```python
# data/download_ecg5000.py
PRIMARY_URL = "http://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"
FALLBACK_URL = "https://www.timeseriesclassification.com/Downloads/ECG5000.zip"
```

Run:

```bash
python data/download_ecg5000.py
```

The intended extracted files are stored under:

```text
data/ECG5000/
```

The dataset files in the repository include:

```text
ECG5000_TRAIN.ts
ECG5000_TEST.ts
ECG5000_TRAIN.arff
ECG5000_TEST.arff
```

This is the exact dataset the project’s loaders and experiment files reference when they train or evaluate ECG5000 experiments.

### 2) CWRU bearing fault dataset (optional/secondary benchmark)

The project also contains scripts for the Case Western Reserve University (CWRU) bearing fault dataset. The preprocessing code identifies the source as:

```text
Source: brjapon/cwru-bearing-datasets on Kaggle
```

The raw data location referenced by the repository script is:

```text
C:\Users\himan\.cache\kagglehub\datasets\brjapon\cwru-bearing-datasets\versions\1\raw
```

and preprocessing is performed by:

```bash
python data/prepare_cwru.py
```

This preprocessing script expects `.mat` files such as:

```text
Time_Normal_1_098.mat
IR007_1_110.mat
IR014_1_175.mat
IR021_1_214.mat
B007_1_123.mat
B014_1_190.mat
B021_1_123.mat
OR007_6_1_136.mat
OR014_6_1_202.mat
OR021_6_1_239.mat
```

and creates the windowed bearing examples used by the repository’s fault classification pipelines.

## Project data layout

The expected folders are:

```text
data/
  ECG5000/
  prepare_cwru.py
  download_ecg5000.py
experiments/
models/
loaders/
```

## Typical setup flow

1. Create and activate a Python virtual environment.
2. Install the packages listed above.
3. Download the ECG5000 archive:

```bash
python data/download_ecg5000.py
```

4. If using the CWRU bearing benchmark, download the Kaggle raw bearing files and run:

```bash
python data/prepare_cwru.py
```

5. Run the experiment or benchmark script you need from `experiments/`.

## Important note

The repository currently contains a large set of generated logs, experiment outputs, and cached artifacts. The `.gitignore` file added to this project is intended to remove editor, Python cache, and environment noise from git status while keeping the source code and data-preparation scripts visible.
