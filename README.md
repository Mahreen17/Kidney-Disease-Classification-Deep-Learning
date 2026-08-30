# Kidney Disease Classification — Deep Learning & MLOps

A production-grade deep learning project for classifying kidney CT scan images into Normal and Tumor categories using a reproducible machine learning pipeline with industry-standard tools and best practices.

**Repository:** [Kidney-Disease-Classification-Deep-Learning](https://github.com/Mahreen17/Kidney-Disease-Classification-Deep-Learning)

---

## Project Overview

This project demonstrates a complete end-to-end machine learning workflow for medical image classification. The pipeline integrates data management, model development, experiment tracking, and cloud deployment to create a production-ready kidney CT scan classification system.

### Key Objectives

- Build a convolutional neural network for binary kidney CT scan classification (Normal vs. Tumor)
- Implement a reproducible ML pipeline using industry-standard DevOps tools
- Track experiments and model versions systematically using MLflow
- Manage datasets and pipeline reproducibility with DVC
- Deploy a lightweight Flask application via Docker on cloud infrastructure
- Optimize the model for resource-constrained deployment environments

---

## Technologies & Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.11 | Programming and scripting |
| **Deep Learning** | TensorFlow / Keras | Model architecture and training |
| **Base Architecture** | VGG16 | Feature extraction from CT images |
| **Model Optimization** | TensorFlow Lite | Lightweight inference for deployment |
| **Experiment Tracking** | MLflow | Parameter, metric, and model versioning |
| **Pipeline Management** | DVC | Data versioning and workflow reproducibility |
| **Configuration** | YAML | Externalized parameter management |
| **Web Framework** | Flask | Prediction API and web interface |
| **Containerization** | Docker | Environment isolation and deployment |
| **Cloud Deployment** | Render | Hosted web application |
| **Version Control** | Git / GitHub | Source code and project management |
| **Development** | Jupyter Notebook | Research and experimentation |

---

## Architecture

### Model Architecture

The project employs VGG16 with ImageNet-pretrained weights as the backbone for feature extraction.

**Input Specifications:**
- Image dimensions: 224 × 224 × 3 (RGB)
- Preprocessing: VGG16 standardized preprocessing function
- Output classes: 2 (Normal, Tumor)

**Network Structure:**

```
Input (224 × 224 × 3)
         ↓
    VGG16 Backbone
    (5 Convolutional Blocks)
         ↓
    Global Average Pooling
         ↓
    Dense Layer (2 units)
         ↓
    Output (Normal / Tumor)
```

**Model Parameters:**
- Total Parameters: 14,715,714
- Trainable Parameters: 12,980,226
- Non-trainable Parameters: 1,735,488

### ML Pipeline Architecture

```
Dataset (Kidney CT Scans)
        ↓
  Data Ingestion
        ↓
  Data Preparation
        ↓
  Base Model Preparation
        ↓
  Model Training
        ↓
  Model Evaluation
        ↓
  MLflow Tracking
        ↓
  model.h5 (155 MB)
        ↓
  TensorFlow Lite Conversion
        ↓
  model.tflite (56 MB)
        ↓
  Flask Prediction Pipeline
        ↓
  Docker Containerization
        ↓
  Render Cloud Deployment
```

---

## Dataset Structure

The dataset is organized into two classification categories:

```
artifacts/
└── data_ingestion/
    └── kidney-ct-scan-image/
        ├── Normal/
        │   └── [CT scan images]
        └── Tumor/
            └── [CT scan images]
```

**Dataset Split:**
- Training: 80%
- Validation: 20%

**Class Mapping:**
- Class 0 → Normal
- Class 1 → Tumor

---

## Image Preprocessing & Augmentation

### Preprocessing

Images are standardized using VGG16's preprocessing function:
- Resized to 224 × 224 pixels
- Applied VGG16-specific normalization
- Consistent preprocessing during training and inference

### Data Augmentation (Training Only)

When enabled, the training dataset applies the following transformations:
- Rotation: 20 degrees
- Horizontal flipping
- Width shifting: 10%
- Height shifting: 10%
- Shearing: 10%
- Zoom: 10%

Augmentation improves model generalization by exposing the network to varied image representations. Validation images remain unaugmented to provide unbiased performance estimates.

---

## Project Structure

```
Kidney-Disease-Classification-Deep-Learning/
│
├── .dvc/                          # DVC configuration
├── .github/workflows/
│   └── main.yaml                  # GitHub Actions CI/CD
│
├── artifacts/
│   └── data_ingestion/
│       └── kidney-ct-scan-image/
│           ├── Normal/
│           └── Tumor/
│
├── config/
│   └── config.yaml                # Project configuration
│
├── model/
│   ├── model.h5                   # Trained TensorFlow model
│   └── model.tflite               # Optimized TensorFlow Lite model
│
├── research/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   └── cnnClassifier/
│       ├── components/            # ML pipeline components
│       ├── config/                # Configuration loaders
│       ├── constants/             # Constants and paths
│       ├── entity/                # Data structures
│       ├── pipeline/
│       │   └── prediction.py      # Prediction pipeline
│       └── utils/                 # Utility functions
│
├── templates/
│   └── index.html                 # Web interface
│
├── app.py                         # Flask application entry point
├── config.yaml                    # General configuration
├── params.yaml                    # Model and training parameters
├── dvc.yaml                       # DVC pipeline definition
├── requirements.txt               # Python dependencies
├── setup.py                       # Package configuration
├── Dockerfile                     # Docker configuration
├── convert_model.py               # TensorFlow Lite conversion script
├── test_tflite.py                 # TensorFlow Lite testing
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

---

## Installation & Setup

### Prerequisites

- Python 3.11
- Git
- pip (Python package manager)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Mahreen17/Kidney-Disease-Classification-Deep-Learning.git
   cd Kidney-Disease-Classification-Deep-Learning
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv kidney
   ```

3. **Activate Virtual Environment**
   
   **Windows:**
   ```bash
   kidney\Scripts\activate
   ```
   
   **Linux/macOS:**
   ```bash
   source kidney/bin/activate
   ```

4. **Upgrade pip**
   ```bash
   python -m pip install --upgrade pip
   ```

5. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the Application**
   ```bash
   python app.py
   ```

The application will be accessible at `http://127.0.0.1:8080`

---

## Configuration Management

### config.yaml

Contains general project configuration including paths for:
- Data ingestion directories
- Model storage locations
- Artifact paths
- Training output directories

### params.yaml

Stores model and training hyperparameters separately from source code:

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

This separation enables experimentation without modifying Python code.

### dvc.yaml

Defines the reproducible ML pipeline stages:
1. Data Ingestion
2. Prepare Base Model
3. Model Training
4. Model Evaluation

---

## Model Training & Evaluation

### Training Pipeline

The training process follows a standard supervised learning workflow:

1. **Data Loading:** Images loaded from directories with automatic class assignment
2. **Preprocessing:** Images resized and normalized per VGG16 specifications
3. **Augmentation:** Training data augmented to improve generalization
4. **Model Training:** Model trained on augmented training data
5. **Validation:** Performance evaluated on held-out validation set
6. **Tracking:** Metrics and parameters logged to MLflow

### Running the DVC Pipeline

To reproduce the entire ML pipeline:

```bash
# Initialize DVC (if not already done)
dvc init

# Reproduce all pipeline stages
dvc repro
```

---

## Model Optimization & Deployment

### TensorFlow Lite Conversion

The original trained model was converted to TensorFlow Lite for efficient deployment:

**Size Reduction:**
- Original H5 model: ~155 MB (162,805,096 bytes)
- TensorFlow Lite model: ~56 MB (58,871,892 bytes)
- Size reduction: ~64%

**Conversion Process:**
```
model.h5 → TensorFlow Lite Converter → model.tflite
```

### Inference Pipeline

TensorFlow Lite inference follows these steps:

```
Input Image
    ↓
Resize to 224 × 224
    ↓
VGG16 Preprocessing
    ↓
Add Batch Dimension
    ↓
TensorFlow Lite Interpreter
    ↓
Prediction Probabilities
    ↓
Argmax (Class Selection)
    ↓
Output (Normal / Tumor)
```

### Model Testing

The TensorFlow Lite model was independently validated:

**Normal Image Test:**
- Predicted Class: 0
- Prediction: Normal

**Tumor Image Test:**
- Predicted Class: 1
- Prediction: Tumor

Results confirm that TensorFlow Lite conversion preserved the original model's prediction behavior.

---

## Flask Application

### API Routes

**GET /**: Displays the web interface with image upload form

**POST /predict**: Prediction endpoint
- Accepts: Multipart form-data with image file
- Returns: JSON with predicted class and label
- Processing: Image resizing, preprocessing, and inference via TensorFlow Lite

**GET/POST /train**: Triggers the training pipeline (development use)

### Application Features

- Image upload interface with file validation
- Real-time prediction processing
- RESTful prediction API
- CORS support for cross-origin requests
- Responsive web interface

### Application Dashboard

![Kidney Disease Classification Dashboard](https://github.com/Mahreen17/Kidney-Disease-Classification-Deep-Learning/blob/main/Screenshot%202026-08-29%20162038.png)

The deployed application provides a user-friendly interface for uploading kidney CT scan images and receiving classification predictions. Users can select an image file, submit it for analysis, and receive immediate results indicating whether the scan shows normal kidney tissue or a tumor.

---

## Docker Deployment

### Docker Configuration

The Dockerfile packages:
- Python 3.11 runtime environment
- Project source code
- Trained TensorFlow Lite model
- Python dependencies from requirements.txt
- Flask application configuration

### Deployment Workflow

```
GitHub Repository
        ↓
    Dockerfile
        ↓
   Docker Build
        ↓
   Container Image
        ↓
    Render Deployment
        ↓
  Live Web Application
```

### Running Locally with Docker

```bash
# Build Docker image
docker build -t kidney-classifier .

# Run container
docker run -p 8080:8080 kidney-classifier
```

---

## Cloud Deployment (Render)

### Render Configuration

The application is deployed as a Docker Web Service on Render.

**Environment Variables:**
- PORT: Dynamically set by Render (defaults to 8080)

**Server Binding:**
- Flask binds to 0.0.0.0 to allow external access

**Deployment Steps:**
1. Connect GitHub repository to Render
2. Configure Docker Web Service
3. Set environment variables
4. Deploy using Dockerfile configuration

---

## Experiment Tracking with MLflow

MLflow provides systematic experiment tracking and management:

**Tracked Components:**
- Training parameters
- Model metrics (accuracy, loss)
- Experiment metadata
- Model artifacts
- Training logs

**Workflow:**
```
Training → Parameters & Metrics → MLflow → Model Registry → Deployment
```

This enables:
- Comparison of different experiment runs
- Systematic hyperparameter tuning
- Model version management
- Reproducible research tracking

---

## Data Version Control with DVC

DVC manages large datasets and ensures pipeline reproducibility:

**Managed Components:**
- Raw dataset versioning
- Processed data versioning
- Pipeline stage outputs
- Model artifacts

**Workflow:**
```
Dataset → DVC Tracking → Pipeline Definition → dvc.yaml → dvc repro
```

Git handles source code versioning; DVC handles data and model versioning for complete project reproducibility.

---

## Reproducibility

This project emphasizes reproducibility through integrated tools and practices:

**Reproducibility Stack:**
- Git: Source code version control
- DVC: Data and pipeline versioning
- MLflow: Experiment tracking
- YAML Configuration: Parameterized experiments
- Python venv: Isolated environment
- Docker: Containerized deployment

**Reproducibility Benefits:**
- Complete pipeline re-execution with `dvc repro`
- Experiment comparison via MLflow
- Environment consistency via Docker
- Parameter modification without code changes
- Git-tracked versioning for all components

---

## Model Performance & Results

### Classification Performance

The model was validated on held-out test data with the following results:

**Normal Images:** Correctly classified with high confidence
**Tumor Images:** Correctly classified with high confidence

### Model Characteristics

- Architecture: VGG16 with transfer learning
- Training approach: Fine-tuning on kidney CT dataset
- Validation methodology: 80/20 train-validation split
- Inference latency: Real-time prediction via TensorFlow Lite

---

## Future Improvements

Potential enhancements to the project:

**Model Enhancement:**
- Increase training epochs for improved convergence
- Implement comprehensive hyperparameter tuning
- Experiment with alternative architectures (ResNet, EfficientNet, MobileNet)
- Add architectural comparison and benchmarking

**Evaluation & Metrics:**
- Implement confusion matrix visualization
- Calculate precision, recall, and F1-score
- Generate ROC-AUC curves
- Create detailed performance reports

**User Experience:**
- Enhance frontend design and responsiveness
- Add prediction confidence score visualization
- Implement image validation and error handling
- Create result history tracking

**Engineering & DevOps:**
- Implement automated testing suite
- Configure CI/CD pipeline with GitHub Actions
- Improve model version management
- Optimize TensorFlow Lite further for edge deployment
- Add API documentation with Swagger/OpenAPI

**Dataset & Domain:**
- Expand dataset size for improved generalization
- Classify additional kidney disease categories
- Improve image validation before prediction
- Collect diverse imaging device data

---

## Limitations

The model operates within specific constraints:

**Classification Scope:**
- Limited to binary classification (Normal vs. Tumor)
- Trained on specific CT scan datasets
- May not generalize to all imaging devices or protocols

**Input Constraints:**
- Designed for kidney CT scan images
- Sensitive to image quality and format
- Performance depends on dataset similarity

**Medical Application:**
- Should not be assumed to generalize to all medical imaging scenarios
- Performance may degrade on images differing significantly from training data
- Not certified for clinical diagnostic use

---

## Medical Disclaimer

This project is intended for **educational and research purposes only**.

This system is not a certified medical diagnostic tool and should not be relied upon for clinical decision-making. Predictions from this application:
- Are not equivalent to professional medical diagnosis
- Should not substitute for qualified healthcare provider evaluation
- Require independent clinical validation before any medical application

**Proper medical diagnosis requires consultation with qualified healthcare professionals.**

---

## Getting Started with Code

### Research Notebooks

The `research/` directory contains Jupyter notebooks documenting the development process:

- `01_data_ingestion.ipynb`: Dataset preparation and organization
- `02_prepare_base_model.ipynb`: VGG16 model preparation and architecture design
- `03_model_training.ipynb`: Training process and evaluation

These notebooks provide detailed explanations of each pipeline stage.

### Quick Start

1. Install dependencies from installation section
2. Run application: `python app.py`
3. Navigate to `http://localhost:8080`
4. Upload kidney CT scan image
5. Receive prediction (Normal or Tumor)

---

## Version Control & Git Workflow

### Git Commands

```bash
# Check status
git status

# Stage changes
git add .

# Commit changes
git commit -m "Descriptive commit message"

# Push to remote
git push origin main
```

### Repository Structure

All project files are maintained in the GitHub repository:
[https://github.com/Mahreen17/Kidney-Disease-Classification-Deep-Learning](https://github.com/Mahreen17/Kidney-Disease-Classification-Deep-Learning)

This includes:
- Source code and modules
- Configuration files (config.yaml, params.yaml)
- Docker configuration
- Research notebooks
- TensorFlow Lite model
- Application templates

---

## Project Status

**Completed Components:**

- Dataset Management and Ingestion
- Data Preparation and Preprocessing
- Model Preparation (VGG16 transfer learning)
- Model Training and Validation
- Model Evaluation
- MLflow Integration
- DVC Pipeline Configuration
- Flask Web Application
- Image Upload Interface
- Prediction Pipeline
- TensorFlow Lite Model Conversion
- Docker Configuration
- GitHub Repository
- Render Cloud Deployment

---

## Author

**Mahreen Begum**  
B.Tech — Artificial Intelligence & Data Science

---

## Support & Contributions

For issues, questions, or contributions, please refer to the GitHub repository.

---

*Last Updated: 2026*
