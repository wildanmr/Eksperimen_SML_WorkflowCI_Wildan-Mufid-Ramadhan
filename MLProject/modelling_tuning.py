import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import dagshub
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Set matplotlib backend untuk CI environment
import matplotlib
matplotlib.use('Agg')

def setup_dagshub_mlflow():
    """Setup DagsHub and MLflow with proper authentication"""
    try:
        # Get credentials from environment variables (for CI)
        username = os.getenv('DAGSHUB_USERNAME') or os.getenv('MLFLOW_TRACKING_USERNAME')
        password = os.getenv('DAGSHUB_PASSWORD') or os.getenv('MLFLOW_TRACKING_PASSWORD')
        
        if username and password:
            # Set DagsHub credentials
            os.environ['MLFLOW_TRACKING_USERNAME'] = username
            os.environ['MLFLOW_TRACKING_PASSWORD'] = password
            print(f"✅ Using DagsHub credentials for user: {username}")
        
        # Initialize DagsHub
        dagshub.init(repo_owner="wildanmr", repo_name="SMSML_Wildan-Mufid-Ramadhan", mlflow=True)
        
        # Set tracking URI
        tracking_uri = "https://dagshub.com/wildanmr/SMSML_Wildan-Mufid-Ramadhan.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Test connection
        try:
            experiments = mlflow.search_experiments()
            print(f"✅ Successfully connected to MLflow at: {tracking_uri}")
            print(f"✅ Found {len(experiments)} existing experiments")
        except Exception as e:
            print(f"⚠️  Warning: Could not list experiments: {e}")
            print("Make sure you're authenticated with DagsHub")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up DagsHub/MLflow: {e}")
        print("Please check your DagsHub credentials and repository access")
        return False

def setup_local_mlflow():
    """Setup local MLflow tracking as fallback"""
    mlflow.set_tracking_uri("file:./mlruns")
    print("✅ Using local MLflow tracking")
    return True

def setup_experiment(experiment_name):
    """Setup MLflow experiment properly"""
    try:
        # Try to get existing experiment first
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Create new experiment if it doesn't exist
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✅ Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            print(f"✅ Using existing experiment: {experiment_name} (ID: {experiment.experiment_id})")
            experiment_id = experiment.experiment_id
        
        # Set the experiment as active
        mlflow.set_experiment(experiment_name)
        
        # Verify the experiment is set correctly
        current_experiment = mlflow.get_experiment_by_name(experiment_name)
        print(f"✅ Current active experiment: {current_experiment.name}")
        
        return experiment_id
        
    except Exception as e:
        print(f"❌ Error setting up experiment: {e}")
        raise

def load_and_prepare_data():
    """Load preprocessed data"""
    try:
        df = pd.read_csv('diabetes_preprocessed.csv')
        print(f"✅ Data loaded successfully: {df.shape}")
        
        # Pisahkan features dan target
        X = df.drop('Diabetes_binary', axis=1)
        y = df['Diabetes_binary']
        
        print(f"✅ Features: {X.shape[1]}, Samples: {len(X)}")
        print(f"✅ Target distribution: {y.value_counts().to_dict()}")
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
    except FileNotFoundError:
        print("❌ Error: 'diabetes_preprocessed.csv' not found")
        print("Please make sure the file exists in the current directory")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def calculate_additional_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate additional metrics beyond autolog"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Additional metrics for Advance (4 pts)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # ROC AUC and Log Loss
    if y_pred_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score, log_loss
            if len(np.unique(y_true)) > 2:
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='weighted')
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        except Exception as e:
            print(f"Warning: Could not calculate ROC AUC or Log Loss: {e}")
    
    # Custom metrics
    unique_classes = len(np.unique(y_true))
    metrics['num_classes'] = unique_classes
    metrics['balanced_accuracy'] = (metrics['recall_weighted'] + metrics['precision_weighted']) / 2
    
    from sklearn.metrics import balanced_accuracy_score
    metrics['sklearn_balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    return metrics

def create_confusion_matrix_plot(y_true, y_pred, model_name, classes=None):
    """Create and save confusion matrix plot"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        
        if classes is None:
            classes = ['No Diabetes', 'Diabetes']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        os.makedirs('artifacts', exist_ok=True)
        plot_path = f'artifacts/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved to: {plot_path}")
        return plot_path
    except Exception as e:
        print(f"Warning: Could not create confusion matrix plot: {e}")
        return None

def create_feature_importance_plot(model, feature_names=None, model_name="Model"):
    """Create and save feature importance plot"""
    try:
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
            
            plt.figure(figsize=(12, 8))
            indices = np.argsort(feature_importance)[::-1]
            
            top_n = min(20, len(feature_importance))
            plt.bar(range(top_n), feature_importance[indices[:top_n]])
            plt.title(f'Top {top_n} Feature Importance - {model_name}')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
            plt.tight_layout()
            
            os.makedirs('artifacts', exist_ok=True)
            plot_path = f'artifacts/feature_importance_{model_name.lower().replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Feature importance plot saved to: {plot_path}")
            return plot_path
    except Exception as e:
        print(f"Warning: Could not create feature importance plot: {e}")
    return None

def save_model_locally(model, model_name="random_forest"):
    """Save model locally with timestamp"""
    try:
        os.makedirs('saved_models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'saved_models/{model_name}_model_{timestamp}.pkl'
        joblib.dump(model, model_filename)
        print(f"✅ Model saved locally to: {model_filename}")
        return model_filename
    except Exception as e:
        print(f"❌ Error saving model locally: {e}")
        return None

def log_model_safely(model, X_train, use_dagshub=True):
    """Safely log model to MLflow with multiple fallback strategies"""
    model_logged = False
    model_path = None
    
    # Strategy 1: Try standard mlflow.sklearn.log_model
    if not model_logged:
        try:
            print("🔄 Attempting to log model with standard MLflow method...")
            input_example = X_train.head(1) if hasattr(X_train, 'head') else X_train[:1]
            
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="best_model",
                input_example=input_example,
                registered_model_name=None
            )
            print(f"✅ Model logged successfully with standard method!")
            print(f"   Model URI: {model_info.model_uri}")
            model_logged = True
            model_path = model_info.model_uri
            
        except Exception as e:
            print(f"⚠️  Standard model logging failed: {e}")
    
    # Strategy 2: Try logging without input_example
    if not model_logged:
        try:
            print("🔄 Attempting to log model without input_example...")
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="best_model",
                registered_model_name=None
            )
            print(f"✅ Model logged successfully without input_example!")
            print(f"   Model URI: {model_info.model_uri}")
            model_logged = True
            model_path = model_info.model_uri
            
        except Exception as e:
            print(f"⚠️  Model logging without input_example failed: {e}")
    
    # Strategy 3: Try using pickle format
    if not model_logged:
        try:
            print("🔄 Attempting to log model as pickle artifact...")
            
            # Save model temporarily as pickle
            temp_model_path = "best_model.pkl"
            joblib.dump(model, temp_model_path)
            
            # Log as artifact
            mlflow.log_artifact(temp_model_path, "model")
            
            # Clean up temp file
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
            
            print("✅ Model logged as pickle artifact!")
            model_logged = True
            
        except Exception as e:
            print(f"⚠️  Pickle artifact logging failed: {e}")
    
    # Strategy 4: Save locally only
    if not model_logged:
        print("🔄 All MLflow model logging methods failed, saving locally only...")
        local_path = save_model_locally(model, "ci_random_forest")
        if local_path:
            print("✅ Model saved locally as fallback")
            model_path = local_path
    
    return model_logged, model_path

def train_model_with_tuning(use_dagshub=True):
    """Train model with hyperparameter tuning - simplified for CI"""
    
    try:
        # Setup tracking - try DagsHub first, fallback to local
        if use_dagshub and setup_dagshub_mlflow():
            print("✅ Using DagsHub MLflow tracking")
        else:
            print("⚠️  Using local MLflow tracking")
            setup_local_mlflow()
        
        # Load data
        X_train, X_test, y_train, y_test = load_and_prepare_data()
        feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        # Create experiment with a fixed name
        experiment_name = "CI_ML_Experiment"
        experiment_id = setup_experiment(experiment_name)
        
        # Disable autolog to prevent duplicate models
        mlflow.sklearn.autolog(disable=True)
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"CI_RandomForest_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            print(f"✅ Started MLflow run: {run.info.run_id}")
            print(f"✅ Experiment ID: {run.info.experiment_id}")
            
            # Simplified hyperparameter tuning for CI (faster)
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            }
            
            # Log hyperparameter grid
            mlflow.log_params({
                "param_grid_n_estimators": str(param_grid['n_estimators']),
                "param_grid_max_depth": str(param_grid['max_depth']),
                "param_grid_min_samples_split": str(param_grid['min_samples_split']),
                "cv_folds": 3,
                "scoring": "accuracy",
                "model_type": "RandomForestClassifier",
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_features": len(X_train.columns) if hasattr(X_train, 'columns') else X_train.shape[1],
                "timestamp": datetime.now().isoformat(),
                "environment": "CI"
            })
            
            # Grid Search
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, scoring='accuracy', 
                n_jobs=-1, verbose=1
            )
            
            print("🔄 Starting hyperparameter tuning...")
            grid_search.fit(X_train, y_train)
            print("✅ Hyperparameter tuning completed")
            
            # Best model
            best_model = grid_search.best_estimator_
            
            # Log best parameters
            for param_name, param_value in grid_search.best_params_.items():
                mlflow.log_param(f"best_{param_name}", param_value)
            
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            # Make predictions
            print("🔄 Making predictions...")
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)
            
            # Calculate all metrics
            metrics = calculate_additional_metrics(y_test, y_pred, y_pred_proba)
            
            # Log all metrics manually
            print("🔄 Logging metrics...")
            for metric_name, metric_value in metrics.items():
                if not np.isnan(metric_value) and not np.isinf(metric_value):
                    mlflow.log_metric(metric_name, float(metric_value))
            
            # Cross-validation scores
            try:
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=3)
                mlflow.log_metric("cv_mean", float(cv_scores.mean()))
                mlflow.log_metric("cv_std", float(cv_scores.std()))
                mlflow.log_metric("cv_min", float(cv_scores.min()))
                mlflow.log_metric("cv_max", float(cv_scores.max()))
            except Exception as e:
                print(f"Warning: Could not calculate CV scores: {e}")
            
            # Create and log artifacts
            print("🔄 Creating visualizations...")
            
            # Feature importance plot
            fi_path = create_feature_importance_plot(best_model, feature_names, "CI Random Forest")
            if fi_path and os.path.exists(fi_path):
                try:
                    mlflow.log_artifact(fi_path)
                    print("✅ Feature importance plot logged to MLflow")
                except Exception as e:
                    print(f"⚠️  Warning: Could not log feature importance to MLflow: {e}")
            
            # Confusion matrix
            cm_path = create_confusion_matrix_plot(y_test, y_pred, "CI Random Forest")
            if cm_path and os.path.exists(cm_path):
                try:
                    mlflow.log_artifact(cm_path)
                    print("✅ Confusion matrix logged to MLflow")
                except Exception as e:
                    print(f"⚠️  Warning: Could not log confusion matrix to MLflow: {e}")
            
            # Classification report
            try:
                report = classification_report(y_test, y_pred)
                os.makedirs('artifacts', exist_ok=True)
                report_path = 'artifacts/classification_report_ci.txt'
                with open(report_path, 'w') as f:
                    f.write(report)
                    f.write(f"\n\nTraining completed at: {datetime.now().isoformat()}")
                    f.write(f"\nEnvironment: CI/CD Pipeline")
                    f.write(f"\nRun ID: {run.info.run_id}")
                mlflow.log_artifact(report_path)
                print("✅ Classification report logged to MLflow")
            except Exception as e:
                print(f"Warning: Could not create/log classification report: {e}")
            
            # Log model with multiple fallback strategies
            print("🔄 Logging model to MLflow...")
            model_logged, model_path = log_model_safely(best_model, X_train, use_dagshub)
            
            if not model_logged:
                print("❌ All model logging strategies failed!")
                # Save locally as final fallback
                local_path = save_model_locally(best_model, "ci_fallback_random_forest")
                if local_path:
                    model_path = local_path
            
            # Log model path as a parameter for reference
            if model_path:
                mlflow.log_param("model_save_path", model_path)
            
            # Log CI-specific information
            mlflow.log_param("ci_run", True)
            mlflow.log_param("github_run_number", os.getenv('GITHUB_RUN_NUMBER', 'unknown'))
            mlflow.log_param("github_ref", os.getenv('GITHUB_REF', 'unknown'))
            mlflow.log_param("github_sha", os.getenv('GITHUB_SHA', 'unknown'))
            
            # Print summary
            print("\n" + "="*60)
            print("🎉 CI TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            print(f"Test accuracy: {metrics['accuracy']:.4f}")
            print(f"MLflow Run ID: {run.info.run_id}")
            if model_path:
                print(f"Model saved at: {model_path}")
            
            # Check if we can get experiment info
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment:
                    print(f"Experiment ID: {experiment.experiment_id}")
                    if use_dagshub:
                        print(f"DagsHub MLflow URL: https://dagshub.com/wildanmr/SMSML_Wildan-Mufid-Ramadhan.mlflow/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}")
                    else:
                        print(f"Local MLflow UI: http://localhost:5000")
            except Exception as e:
                print(f"Could not get experiment info: {e}")
            
            print("="*60)
            
            return best_model, metrics, run.info.run_id
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        # Exit with error code for CI
        raise e

if __name__ == "__main__":
    print("🚀 Starting CI MLflow Training...")
    print("="*60)
    
    # Always use DagsHub in CI
    USE_DAGSHUB = True
    
    try:
        model, metrics, run_id = train_model_with_tuning(use_dagshub=USE_DAGSHUB)
        print("\n✅ CI Training completed successfully!")
        
        print("🔗 Check your DagsHub repository for MLflow tracking results:")
        print("   https://dagshub.com/wildanmr/SMSML_Wildan-Mufid-Ramadhan.mlflow")
        
        # Create summary file for CI
        summary_path = 'training_summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Training Summary - {datetime.now().isoformat()}\n")
            f.write("="*50 + "\n")
            f.write(f"MLflow Run ID: {run_id}\n")
            f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}\n")
            f.write(f"Precision (Weighted): {metrics['precision_weighted']:.4f}\n")
            f.write(f"Recall (Weighted): {metrics['recall_weighted']:.4f}\n")
            f.write("\nTraining completed successfully in CI environment!\n")
        
        print(f"✅ Training summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"\n❌ CI Training failed: {e}")
        print("Please check your data path, column names, and authentication.")
        # Create error log for CI
        error_path = 'training_error.txt'
        with open(error_path, 'w') as f:
            f.write(f"Training Error - {datetime.now().isoformat()}\n")
            f.write("="*50 + "\n")
            f.write(f"Error: {str(e)}\n")
            f.write("\nPlease check the logs for more details.\n")
        
        # Exit with error code for CI
        exit(1)