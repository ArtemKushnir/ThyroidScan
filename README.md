# ThyroidScan
[![CI](https://github.com/ArtemKushnir/ThyroidScan/actions/workflows/main.yaml/badge.svg)](https://github.com/ArtemKushnir/ThyroidScan/actions/workflows/main.yaml)


A medical tool for classifying and segmenting thyroid diseases based on medical images.

## Project Overview
Thyroid Scan is an intelligent diagnostic system that uses machine learning to analyze thyroid medical images. The project is designed to assist medical professionals in accurate and rapid classification of  pathologies.

## Technology Stack
- **Programming Language:** Python 3.8+
- **Deep Learning Framework:** PyTorch
- **Machine Learning:** scikit-learn
- **Optimization:** Optuna
- **Medical Image Processing:** MONAI
- **Data Analysis:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM (16GB+ recommended)

### Clone Repository
<pre><code>git clone https://github.com/ArtemKushnir/ThyroidScan.git
cd ThyroidScan</code></pre>

### Install Dependencies
**Core dependencies**
<pre><code>pip install -r requirements.txt</code></pre>

**Development dependencies**
<pre><code>pip install -r requirements.dev.txt</code></pre>

## Quick Start
### Basic training example
```python
import src.training_module.main_pipeline.training_system

from src.data_loaders.ddti_loader import DDTILoader
from src.data_loaders.bus_bra_loader import BUSLoader

DATA_PATH = "path to BUS-BRA dataset"
bus_loader = BUSLoader(DATA_PATH)

XML_PATH = "path to DDTI dataset xml folder"
IMAGE_PATH = "path to DDTI dataset image folder"
ddti_loader = DDTILoader(XML_PATH, IMAGE_PATH)

config = training_system.Config(
    loader=ddti_loader,
    models=["svm"],
    experiment_dir="experiments",
    target_metric="f1",
    tune_params=False,
    is_update=False,
)

ml_system = training_system.MLSystemFacade(config)

# Use-case 1: train single model
model_name = ml_system.train_single_model("svm")

# Use-case 2: train all models
models = ml_system.train_all_models()

# Use-case 3: model comparison
exp_id = ml_system.run_model_comparison_experiment("experiment_name", save_plots=True)
```
