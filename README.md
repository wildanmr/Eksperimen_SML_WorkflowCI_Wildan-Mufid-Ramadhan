# Diabetes Prediction ML Pipeline
Advance automated ML pipeline for diabetes prediction using MLflow, GitHub Actions, and Docker Hub.

## 🎯 Features

* ✅ **Automated model training** with hyperparameter tuning using RandomForest
* ✅ **MLflow tracking** with DagsHub integration
* ✅ **GitHub Actions CI/CD pipeline** for automated training
* ✅ **Docker containerization** with MLflow model serving
* ✅ **Artifact storage** via GitHub Actions and Docker Hub
* ✅ **Advanced metrics** calculation and visualization

## 📁 Project Structure

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml          # GitHub Actions workflow
├── MLProject/
│   ├── MLproject                    # MLflow project config
│   ├── conda.yaml                   # Conda environment
│   ├── modelling.py                 # Main training script
│   ├── requirements.txt            # Python dependencies
│   └── Dockerfile                  # Docker configuration
├── README.md
└── .gitignore
```

## 🔄 CI/CD Pipeline

### Pipeline Steps:

1. **Model Training**: Hyperparameter tuning with GridSearch
2. **Metrics Calculation**: Advanced metrics and visualizations
3. **MLflow Logging**: Model, metrics, and artifacts to DagsHub
4. **Artifact Storage**: Upload to GitHub Actions artifacts
5. **Docker Build**: Build and push to Docker Hub
6. **MLflow Docker**: Build serving image using `mlflow build-docker`

## 📈 Monitoring

* **MLflow UI**: [https://dagshub.com/wildanmr/SMSML\_Wildan-Mufid-Ramadhan.mlflow](https://dagshub.com/wildanmr/SMSML_Wildan-Mufid-Ramadhan.mlflow)
* **GitHub Actions**: Repository > Actions tab
* **Docker Images**: Docker Hub repository

## 🐳 Docker Images

Once the pipeline completes, the following Docker images are available:

* `wildanmr/diabetes-ml-model:latest` - Standard training image
* `wildanmr/diabetes-ml-model:<build-number>` - Versioned training image
* `wildanmr/diabetes-ml-mlflow:latest` - MLflow serving image

### Run Serving Container

```bash
docker pull wildanmr/diabetes-ml-mlflow:latest
docker run -p 5000:5000 wildanmr/diabetes-ml-mlflow:latest
```

## 📋 Model Details

* **Algorithm**: Random Forest Classifier
* **Hyperparameters**: Grid search over n\_estimators, max\_depth, min\_samples\_split, min\_samples\_leaf
* **Dataset**: Diabetes dataset (automatically downloaded from GitHub releases)
* **Metrics**: Accuracy, Precision, Recall, F1-score (weighted, macro, micro), ROC-AUC, Log Loss
