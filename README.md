# Diabetes Prediction ML Pipeline

Advance automated ML pipeline for diabetes prediction using MLflow, GitHub Actions, Docker Hub, and GitHub Container Registry (GHCR).

## 🎯 Features

* ✅ **Automated model training** with hyperparameter tuning using RandomForest
* ✅ **MLflow tracking** with DagsHub integration
* ✅ **GitHub Actions CI/CD pipeline** for automated training
* ✅ **Docker containerization** with MLflow model serving
* ✅ **Artifact storage** via GitHub Actions, Docker Hub, and GitHub Releases
* ✅ **Advanced metrics** calculation and visualization

## 📁 Project Structure

```text
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml          # GitHub Actions workflow
├── MLProject/
│   ├── MLproject                    # MLflow project config
│   ├── conda.yaml                   # Conda environment
│   ├── modelling.py                 # Main training script
│   ├── requirements.txt             # Python dependencies
│   └── Dockerfile                   # Docker configuration
├── README.md
└── .gitignore
```

## 🔄 CI/CD Pipeline

### Pipeline Steps:

1. **Model Training**: Hyperparameter tuning with GridSearch
2. **Metrics Calculation**: Advanced metrics and visualizations
3. **MLflow Logging**: Model, metrics, and artifacts to DagsHub
4. **Artifact Storage**: Upload to GitHub Actions artifacts and GitHub Releases
5. **Docker Build**: Build and push to Docker Hub **and** GHCR
6. **MLflow Docker**: Build serving image using `mlflow build-docker`

## 📈 Monitoring

* **MLflow UI**: [https://dagshub.com/wildanmr/SMSML\_Wildan-Mufid-Ramadhan.mlflow](https://dagshub.com/wildanmr/SMSML_Wildan-Mufid-Ramadhan.mlflow)
* **GitHub Actions**: Repository → **Actions** tab
* **GitHub Releases**: Download final artifacts from the **Releases** page
* **Docker Images**: Docker Hub & GHCR repositories

## 🐳 Docker Images

After the pipeline succeeds, pre-built images are published to both Docker Hub and GitHub Container Registry (GHCR):

| Registry   | Image                                        |
| ---------- | -------------------------------------------- |
| Docker Hub | `wildanmr/diabetes-ml-mlflow:latest`         |
| GHCR       | `ghcr.io/wildanmr/diabetes-ml-mlflow:latest` |

### Pull & Run the Serving Container

```bash
# Pull from Docker Hub
docker pull wildanmr/diabetes-ml-mlflow:latest

# OR pull from GHCR
docker pull ghcr.io/wildanmr/diabetes-ml-mlflow:latest

# Run the container (choose the image you pulled)
docker run -p 5000:5000 ghcr.io/wildanmr/diabetes-ml-mlflow:latest
```

## 📋 Model Details

* **Algorithm**: Random Forest Classifier
* **Hyperparameters**: Grid search over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
* **Dataset**: Diabetes dataset (automatically downloaded from GitHub Releases)
* **Metrics**: Accuracy, Precision, Recall, F1-score (weighted, macro, micro), ROC-AUC, Log Loss