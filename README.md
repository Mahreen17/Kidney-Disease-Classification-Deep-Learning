# Kidney Disease Classification — MLflow & DVC

A deep learning project for **classifying kidney disease from medical images** using a structured, reproducible machine learning pipeline.

The project integrates **TensorFlow/Keras** for deep learning, **MLflow** for experiment and model tracking, and **DVC (Data Version Control)** for data versioning and pipeline management.

---

## Project Overview

Kidney diseases can be identified and analyzed using medical imaging techniques. This project aims to build a deep learning model capable of classifying kidney-related medical images into their respective categories.

The project follows an **end-to-end Machine Learning pipeline**, starting from data management and preprocessing to model training, evaluation, experiment tracking, and deployment.

### Main Objectives

* Build a deep learning model for kidney disease image classification.
* Create a reproducible ML pipeline.
* Track experiments, parameters, metrics, and models using MLflow.
* Version datasets and ML pipelines using DVC.
* Maintain project configurations separately using YAML files.
* Prepare the trained model for application/deployment.

---

## Technologies Used

| Technology             | Purpose                                |
| ---------------------- | -------------------------------------- |
| Python 3.11            | Programming language                   |
| TensorFlow / Keras     | Deep learning and image classification |
| MLflow                 | Experiment and model tracking          |
| DVC                    | Data and pipeline versioning           |
| YAML                   | Configuration and parameter management |
| Jupyter Notebook       | Research and experimentation           |
| Streamlit / Python App | Model deployment                       |
| Git / GitHub           | Source-code version control            |

---

## Project Structure

```text
Kidney-Disease-Classification/
│
├── .dvc/                    # DVC internal files
├── .github/                 # GitHub-related configuration
│
├── artifacts/               # Generated project artifacts
│
├── config/
│   └── config.yaml          # Project configuration
│
├── research/
│   └── *.ipynb              # Experiments and research notebooks
│
├── src/
│   └── cnnClassifier/
│       ├── components/      # Individual pipeline components
│       ├── config/          # Configuration management
│       ├── constants/       # Project constants
│       ├── entity/          # Configuration entities
│       ├── pipeline/        # ML pipeline stages
│       └── utils/           # Utility functions
│
├── templates/               # Application templates
│
├── app.py                   # Application / deployment entry point
├── config.yaml              # Main configuration file
├── params.yaml              # Model and training parameters
├── dvc.yaml                 # DVC pipeline definition
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── README.md                # Project documentation
└── .gitignore               # Git ignored files
```

> **Note:** The exact folder structure may vary depending on the stages implemented in the project.

---

## Key Configuration Files

### `config.yaml`

Contains the project's general configuration settings, such as:

* Dataset paths
* Model directories
* Artifact locations
* Training-related configuration

### `params.yaml`

Stores model and training parameters separately from the source code.

Example:

```yaml
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 1
CLASSES: 2
WEIGHTS: imagenet
LEARNING_RATE: 0.01
```

Keeping parameters in a separate file makes it easier to experiment with different configurations without modifying the Python source code.

### `dvc.yaml`

Defines the different stages of the machine learning pipeline.

A typical pipeline may contain stages such as:

```text
Data Ingestion
      ↓
Prepare Base Model
      ↓
Model Training
      ↓
Model Evaluation
```

DVC allows these stages to be executed in a reproducible manner.

---

## MLflow

**MLflow** is used to track machine learning experiments.

It can record:

* Parameters
* Training metrics
* Model performance
* Experiments
* Trained models
* Artifacts

This makes it easier to compare different experiments and understand which configuration produces the best model.

### Example MLflow Workflow

```text
Model Training
      ↓
Parameters
      ↓
Metrics
      ↓
MLflow Tracking
      ↓
Experiment Comparison
      ↓
Best Model
```

---

## DVC

**DVC (Data Version Control)** is used to manage datasets and machine learning pipelines.

Git is mainly used for tracking source code, while DVC can track large datasets and generated ML artifacts.

### DVC Workflow

```text
Dataset
   ↓
DVC Tracking
   ↓
Pipeline Definition
   ↓
Model Training
   ↓
Model Evaluation
```

This helps make the project **reproducible** and easier to maintain.

---

## Python Virtual Environment

The original tutorial uses **Anaconda/Conda**, but this project uses Python's built-in **`venv`** instead.

### Create the virtual environment

```bash
python -m venv kidney
```

### Activate the environment — Windows

```bash
kidney\Scripts\activate
```

After activation, the terminal should show:

```text
(kidney)
```

### Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Install project dependencies

```bash
pip install -r requirements.txt
```

---

## Why Python 3.11?

During setup, an issue occurred while installing:

```text
tensorflow==2.12.0
```

The error indicated that the requested TensorFlow version was not compatible with the Python version initially being used.

The environment was therefore configured using a **compatible Python version**, such as Python 3.11.

The recommended setup is:

```text
Python 3.11
     ↓
   venv
     ↓
  kidney
     ↓
requirements.txt
```

---

## Project Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the project

```bash
cd Kidney-Disease-Classification
```

### 3. Create the virtual environment

```bash
python -m venv kidney
```

### 4. Activate the environment

```bash
kidney\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Initialize DVC

```bash
dvc init
```

> If DVC has already been initialized in the repository, do not run `dvc init` again.

### 7. Run the pipeline

```bash
dvc repro
```

### 8. Run the application

```bash
python app.py
```

---

## Machine Learning Pipeline

The overall workflow of the project is:

```text
                    ┌─────────────────┐
                    │     Dataset     │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Data Ingestion  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Data Preparation│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  Base Model     │
                    │  Preparation    │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Model Training  │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Model Evaluation│
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ MLflow Tracking │
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │ Model Deployment│
                    └─────────────────┘
```

DVC manages the reproducibility of the pipeline, while MLflow tracks experiments and model performance.

---

## Experiment Tracking

MLflow can be used to compare multiple training experiments.

For example:

```text
Experiment 1
├── Learning Rate: 0.01
├── Epochs: 10
└── Accuracy: ...

Experiment 2
├── Learning Rate: 0.001
├── Epochs: 20
└── Accuracy: ...

Experiment 3
├── Learning Rate: 0.0001
├── Epochs: 30
└── Accuracy: ...
```

The experiments can then be compared to identify the best-performing configuration.

---

## Reproducibility

One of the main goals of this project is **reproducibility**.

The combination of:

```text
Git
 +
DVC
 +
MLflow
 +
YAML Configuration
 +
Virtual Environment
```

helps ensure that experiments and model training can be repeated consistently.

---

## Expected Outcome

At the end of the project, the system should provide an end-to-end workflow for:

```text
Medical Images
      ↓
Data Processing
      ↓
Deep Learning Model
      ↓
Kidney Disease Classification
      ↓
Model Evaluation
      ↓
MLflow Experiment Tracking
      ↓
DVC Pipeline & Versioning
      ↓
Application Deployment
```

---

## Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Data augmentation
* Improving model architecture
* Increasing classification accuracy
* Adding more evaluation metrics
* Comparing multiple deep learning architectures
* Model versioning with MLflow
* Automated CI/CD
* Cloud deployment
* Improved application UI

---

## Medical Disclaimer

This project is intended **for educational and research purposes only**.

It is not a certified medical diagnostic system and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## Author

**Mahreen Begum**

B.Tech — Artificial Intelligence & Data Science

---

## Project Goal

> **Build a reproducible deep learning pipeline for kidney disease classification while learning industry-standard tools such as MLflow, DVC, Git, and Python virtual environments.**
