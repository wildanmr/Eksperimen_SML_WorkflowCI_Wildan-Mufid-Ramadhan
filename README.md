# 🩺 Diabetes Prediction MLflow CI/CD Pipeline

Advanced CI/CD pipeline for automated diabetes prediction model training, deployment, and Docker containerization using MLflow Projects and GitHub Actions.

## 📁 Project Structure

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI/CD workflow
├── MLProject/
│   ├── modelling.py                  # Updated ML training script with CLI args
│   ├── conda.yaml                    # Environment dependencies
│   ├── MLProject                     # MLflow project configuration
│   ├── predict.py                    # Model serving script
│   ├── Dockerfile                    # Docker configuration
│   ├── diabetes_preprocessed.csv     # Preprocessed dataset
│   └── RUN_SUMMARY.md               # Auto-updated run summaries
└── README.md                        # This file
```

## 🐳 Docker Usage

### Pull and Run Model Container

```bash
# Pull the image
docker pull wildanmr/diabetes-prediction-model:latest

# Run the container
docker run -p 8080:8080 wildanmr/diabetes-prediction-model:latest
```

### API Endpoints

Once running, access these endpoints:

- **`http://localhost:8080/`** - Information page
- **`http://localhost:8080/health`** - Health check
- **`http://localhost:8080/info`** - Model information
- **`http://localhost:8080/predict`** - Make predictions

### Example Prediction Request

**GET Request:**
```
http://localhost:8080/predict?HighBP=1&HighChol=0&CholCheck=1&BMI=25.5&Smoker=0&Stroke=0&HeartDiseaseorAttack=0&PhysActivity=1&Fruits=1&Veggies=1&HvyAlcoholConsump=0&AnyHealthcare=1&NoDocbcCost=0&GenHlth=2&MentHlth=5&PhysHlth=0&DiffWalk=0&Sex=1&Age=8&Education=6&Income=7
```

**POST Request:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "HighBP": 1,
    "HighChol": 0,
    "CholCheck": 1,
    "BMI": 25.5,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 2,
    "MentHlth": 5,
    "PhysHlth": 0,
    "DiffWalk": 0,
    "Sex": 1,
    "Age": 8,
    "Education": 6,
    "Income": 7
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "No Diabetes",
  "probability": {
    "no_diabetes": 0.85,
    "diabetes": 0.15
  },
  "confidence": 0.85,
  "features_used": 21,
  "missing_features": [],
  "timestamp": "2025-06-24T10:30:00.000Z"
}
```

## 🔧 Customization

### Modify Training Parameters

Edit the workflow file (`.github/workflows/ci.yml`) to change default parameters:

```yaml
workflow_dispatch:
  inputs:
    n_estimators:
      default: '300'  # Change default value
    max_depth:
      default: '25'   # Change default value
```