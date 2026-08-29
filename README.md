# Kidney Disease Classification — Deep Learning, MLflow & DVC

A complete end-to-end deep learning project for classifying kidney medical images using **TensorFlow/Keras**, **MLflow** for experiment tracking, **DVC** for data and pipeline management, and **Flask** for model deployment.

[![Live Application](https://img.shields.io/badge/Live%20App-Render-brightgreen)](https://kidney-disease-classification-deep-girb.onrender.com)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/Mahreen17/Kidney-Disease-Classification-Deep-Learning.git)

---

## Project Overview

This project implements a complete **machine learning pipeline** for kidney disease classification using deep learning and medical imaging. The workflow covers:

- Data ingestion and preparation
- Convolutional Neural Network (CNN) model development
- Model training and evaluation
- Experiment tracking with MLflow
- Pipeline management with DVC
- Web-based Flask application
- Docker containerization
- Cloud deployment on Render

**Classification Categories:**
- Normal
- Tumor

---

## Project Objectives

- Build a deep learning model for kidney image classification
- Develop a structured and reproducible ML pipeline
- Implement experiment tracking with MLflow
- Manage data versioning and pipelines with DVC
- Separate configuration from source code using YAML
- Create a Flask web application for inference
- Containerize with Docker
- Deploy to cloud with Render
- Maintain version control with Git and GitHub

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| **Python** | Programming language |
| **TensorFlow/Keras** | Deep learning framework |
| **NumPy** | Numerical operations |
| **Pandas** | Data processing |
| **Matplotlib & Seaborn** | Data visualization |
| **MLflow** | Experiment and model tracking |
| **DVC** | Data versioning and pipeline management |
| **Flask** | Web application framework |
| **Docker** | Application containerization |
| **Render** | Cloud deployment platform |
| **Git & GitHub** | Version control |
| **Jupyter** | Interactive development |

---

## Project Structure

```
Kidney-Disease-Classification-Deep-Learning/
│
├── .dvc/                          # DVC configuration
├── .github/                       # GitHub configuration
│
├── artifacts/                     # Generated artifacts
│   ├── data_ingestion/
│   ├── prepare_base_model/
│   ├── model_training/
│   └── model_evaluation/
│
├── config/
│   └── config.yaml               # Project configuration
│
├── model/
│   └── model.h5                  # Trained model
│
├── research/                      # Experimentation notebooks
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation_with_mlflow.ipynb
│
├── src/cnnClassifier/
│   ├── components/               # Core pipeline components
│   │   ├── data_ingestion.py
│   │   ├── prepare_base_model.py
│   │   ├── model_training.py
│   │   └── model_evaluation_mlflow.py
│   │
│   ├── config/
│   │   └── configuration.py
│   │
│   ├── constants/
│   │   └── __init__.py
│   │
│   ├── entity/
│   │   └── config_entity.py
│   │
│   ├── pipeline/                 # ML pipeline stages
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_model_training.py
│   │   └── stage_04_model_evaluation.py
│   │
│   └── utils/
│       └── common.py
│
├── templates/
│   └── index.html                # Web interface
│
├── app.py                        # Flask application
├── config.yaml                   # Configuration file
├── params.yaml                   # Model parameters
├── dvc.yaml                      # DVC pipeline definition
├── requirements.txt              # Python dependencies
├── setup.py                      # Project setup
├── Dockerfile                    # Docker configuration
├── .gitignore
└── README.md
```

---

## Model Information

### Architecture
- **Type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow/Keras
- **Base Model:** Utilized transfer learning approach

### Input Specifications
- **Image Size:** 224 × 224 pixels
- **Normalization:** `image / 255.0`
- **Classes:** 2 (Normal, Tumor)
- **Output Model:** `model/model.h5`

### Prediction Pipeline
```
Input Image → Load → Resize (224×224) → Convert to Array 
→ Normalize → Add Batch Dimension → CNN Model → Argmax 
→ Prediction (Normal/Tumor)
```

---

## Configuration Management

### `config.yaml`
Stores project-level settings:
- Dataset locations
- Artifact directories
- Model paths
- Training-related paths

### `params.yaml`
Stores model and training parameters:
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

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### 1. Clone Repository
```bash
git clone https://github.com/Mahreen17/Kidney-Disease-Classification-Deep-Learning.git
cd Kidney-Disease-Classification-Deep-Learning
```

### 2. Create Virtual Environment
```bash
python -m venv kidney
```

### 3. Activate Virtual Environment

**On Windows:**
```bash
kidney\Scripts\activate
```

**On macOS/Linux:**
```bash
source kidney/bin/activate
```

### 4. Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 5. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running Locally

### 1. Execute DVC Pipeline
```bash
dvc repro
```
This runs the complete ML pipeline:
- Data Ingestion
- Base Model Preparation
- Model Training
- Model Evaluation

### 2. Launch Flask Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

### 3. Access the Web Interface
- **Home Page:** `GET /`
- **Training:** `GET /train` or `POST /train`
- **Prediction:** `POST /predict`

---

## Docker & Deployment

### Build Docker Image
```bash
docker build -t kidney-disease-classifier .
```

### Run Docker Container
```bash
docker run -p 5000:5000 kidney-disease-classifier
```

### Docker Configuration
- **Base Image:** `python:3.8-slim-buster`
- **Working Directory:** `/app`
- **Entry Point:** `python3 app.py`

### Cloud Deployment (Render)
```
GitHub Repository
    ↓
Dockerfile
    ↓
Render Build
    ↓
Docker Container
    ↓
Flask Application
    ↓
Live Web Application
```

**Live Application:** [Kidney Disease Classification on Render](https://kidney-disease-classification-deep-girb.onrender.com)

---

## DVC Pipeline Management

The DVC pipeline is defined in `dvc.yaml` and follows this workflow:

```
Data Ingestion
    ↓
Prepare Base Model
    ↓
Model Training
    ↓
Model Evaluation
```

**Run entire pipeline:**
```bash
dvc repro
```

**Visualize pipeline:**
```bash
dvc dag
```

---

## MLflow Experiment Tracking

MLflow tracks:
- Experiment runs
- Model parameters
- Training metrics
- Model artifacts
- Evaluation results

**Access MLflow UI:**
```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view experiment dashboard.

### Dashboard

![MLflow Dashboard](https://github.com/Mahreen17/Kidney-Disease-Classification-Deep-Learning/blob/main/Screenshot%202026-08-29%20162038.png)

---

## End-to-End Workflow

```
Medical Image Dataset
    ↓
Data Ingestion (DVC)
    ↓
Data Preparation
    ↓
Base Model Preparation
    ↓
Model Training
    ↓
Model Evaluation
    ├─ DVC (Pipeline Management)
    └─ MLflow (Experiment Tracking)
    ↓
Trained Model (model/model.h5)
    ↓
Prediction Pipeline
    ↓
Flask Web Application
    ↓
Docker Container
    ↓
Render Deployment
    ↓
Live Application
```

---

## API Usage

### Prediction Endpoint
```bash
POST /predict
Content-Type: multipart/form-data

# Upload image file
```

**Response:**
```json
[
    {
        "image": "Normal"
    }
]
```

or

```json
[
    {
        "image": "Tumor"
    }
]
```

---

## Key Dependencies

```
tensorflow==2.12.0
pandas
dvc
mlflow==2.2.2
notebook
numpy
matplotlib
seaborn
python-box==6.0.2
pyYAML
Flask
Flask-Cors
gdown
```

See `requirements.txt` for complete list.

---

## Reproducibility

This project ensures reproducibility through:

| Tool | Purpose |
|------|---------|
| **Git** | Source code version control |
| **GitHub** | Remote repository hosting |
| **DVC** | Dataset and pipeline versioning |
| **MLflow** | Experiment tracking |
| **YAML** | Configuration management |
| **venv** | Environment isolation |
| **Docker** | Containerization |
| **Render** | Consistent deployment |

---

## Future Improvements

- Improve model accuracy and performance
- Expand dataset with more diverse samples
- Implement advanced data augmentation
- Perform hyperparameter tuning
- Compare different CNN architectures
- Add confidence scores to predictions
- Enhance web interface UX/UI
- Implement robust model validation
- Add CI/CD automated workflows
- Use production-grade WSGI server
- Implement comprehensive logging
- Add model versioning and lifecycle management

---

## Medical Disclaimer

**This project is for educational and research purposes only.**

- Not a certified medical diagnostic system
- Should not substitute professional medical advice
- Predictions are not medically validated results
- For research and educational demonstration only

---

## Author

**Mahreen Begum**

B.Tech — Artificial Intelligence & Data Science

---

## Links

- **Live Application:** [Kidney Disease Classification on Render](https://kidney-disease-classification-deep-girb.onrender.com)
- **GitHub Repository:** [Mahreen17/Kidney-Disease-Classification-Deep-Learning](https://github.com/Mahreen17/Kidney-Disease-Classification-Deep-Learning.git)

---

## Acknowledgments

- TensorFlow/Keras documentation
- MLflow project management
- DVC pipeline management
- Render deployment platform
- Open-source community contributions

---

**Project Status:** Complete and Deployed

All components including deep learning model, pipeline, web application, and cloud deployment are fully functional.
