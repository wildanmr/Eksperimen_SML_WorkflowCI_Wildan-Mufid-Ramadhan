import pandas as pd
import numpy as np
import argparse
import sys
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

def parse_arguments():
    """Parse command line arguments for MLflow Project"""
    parser = argparse.ArgumentParser(description='Train diabetes prediction model')
    parser.add_argument('--data_path', type=str, default='diabetes_preprocessed.csv',
                      help='Path to the preprocessed dataset')
    parser.add_argument('--use_dagshub', type=str, default='true',
                      help='Whether to use DagsHub MLflow tracking')
    parser.add_argument('--experiment_name', type=str, default='CI_Auto_Training',
                      help='Name of the MLflow experiment')
    parser.add_argument('--model_name', type=str, default='diabetes_rf_model',
                      help='Name for the saved model')
    parser.add_argument('--max_depth', type=int, default=20,
                      help='Maximum depth for Random Forest')
    parser.add_argument('--n_estimators', type=int, default=200,
                      help='Number of estimators for Random Forest')
    parser.add_argument('--min_samples_split', type=int, default=5,
                      help='Minimum samples required to split a node')
    parser.add_argument('--min_samples_leaf', type=int, default=2,
                      help='Minimum samples required at a leaf node')
    
    return parser.parse_args()

def setup_dagshub_mlflow():
    """Setup DagsHub and MLflow with proper authentication"""
    try:
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
    print("✅ Make sure to run 'mlflow ui' in terminal to view results")
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

def load_and_prepare_data(data_path):
    """Load preprocessed data"""
    try:
        if not os.path.exists(data_path):
            print(f"❌ Error: '{data_path}' not found")
            print(f"Current directory: {os.getcwd()}")
            print(f"Files in current directory: {os.listdir('.')}")
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        print(f"✅ Data loaded successfully: {df.shape}")
        
        # Pisahkan features dan target
        X = df.drop('Diabetes_binary', axis=1)
        y = df['Diabetes_binary']
        
        print(f"✅ Features: {X.shape[1]}, Samples: {len(X)}")
        print(f"✅ Target distribution: {y.value_counts().to_dict()}")
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
    except FileNotFoundError:
        print(f"❌ Error: '{data_path}' not found")
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
    
    # Additional metrics
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

def log_model_safely(model, X_train, model_name="diabetes_rf_model", use_dagshub=True):
    """Safely log model to MLflow with multiple fallback strategies"""
    model_logged = False
    model_uri = None
    
    # Strategy 1: Try standard mlflow.sklearn.log_model
    if not model_logged:
        try:
            print("🔄 Attempting to log model with standard MLflow method...")
            input_example = X_train.head(1) if hasattr(X_train, 'head') else X_train[:1]
            
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=input_example,
                registered_model_name=None
            )
            print(f"✅ Model logged successfully with standard method!")
            print(f"   Model URI: {model_info.model_uri}")
            model_logged = True
            model_uri = model_info.model_uri
            
        except Exception as e:
            print(f"⚠️  Standard model logging failed: {e}")
    
    # Strategy 2: Try logging without input_example
    if not model_logged:
        try:
            print("🔄 Attempting to log model without input_example...")
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=None
            )
            print(f"✅ Model logged successfully without input_example!")
            print(f"   Model URI: {model_info.model_uri}")
            model_logged = True
            model_uri = model_info.model_uri
            
        except Exception as e:
            print(f"⚠️  Model logging without input_example failed: {e}")
    
    # Strategy 3: Try using pickle format
    if not model_logged:
        try:
            print("🔄 Attempting to log model as pickle artifact...")
            
            # Save model temporarily as pickle
            temp_model_path = f"temp_{model_name}.pkl"
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
        local_path = save_model_locally(model, model_name)
        if local_path:
            print("✅ Model saved locally as fallback")
            model_uri = local_path
    
    return model_logged, model_uri

def train_model_with_parameters(args):
    """Train model with specified parameters from command line"""
    
    try:
        # Convert string to boolean for use_dagshub
        use_dagshub = args.use_dagshub.lower() == 'true'
        
        # Setup tracking - try DagsHub first, fallback to local
        if use_dagshub and setup_dagshub_mlflow():
            print("✅ Using DagsHub MLflow tracking")
        else:
            print("⚠️  Using local MLflow tracking")
            setup_local_mlflow()
        
        # Load data
        X_train, X_test, y_train, y_test = load_and_prepare_data(args.data_path)
        feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        
        # Create experiment
        experiment_id = setup_experiment(args.experiment_name)
        
        # Disable autolog to prevent duplicate models
        mlflow.sklearn.autolog(disable=True)
        
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"CI_Training_{datetime.now().strftime('%H%M%S')}") as run:
            print(f"✅ Started MLflow run: {run.info.run_id}")
            print(f"✅ Experiment ID: {run.info.experiment_id}")
            
            # Create model with specified parameters
            model = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_split=args.min_samples_split,
                min_samples_leaf=args.min_samples_leaf,
                random_state=42
            )
            
            # Log hyperparameters
            mlflow.log_params({
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_split": args.min_samples_split,
                "min_samples_leaf": args.min_samples_leaf,
                "model_type": "RandomForestClassifier",
                "train_size": len(X_train),
                "test_size": len(X_test),
                "n_features": len(X_train.columns) if hasattr(X_train, 'columns') else X_train.shape[1],
                "data_path": args.data_path,
                "model_name": args.model_name,
                "timestamp": datetime.now().isoformat()
            })
            
            # Train model
            print("🔄 Training model...")
            model.fit(X_train, y_train)
            print("✅ Model training completed")
            
            # Make predictions
            print("🔄 Making predictions...")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate all metrics
            metrics = calculate_additional_metrics(y_test, y_pred, y_pred_proba)
            
            # Log all metrics manually
            print("🔄 Logging metrics...")
            for metric_name, metric_value in metrics.items():
                if not np.isnan(metric_value) and not np.isinf(metric_value):
                    mlflow.log_metric(metric_name, float(metric_value))
            
            # Cross-validation scores
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3)
                mlflow.log_metric("cv_mean", float(cv_scores.mean()))
                mlflow.log_metric("cv_std", float(cv_scores.std()))
                mlflow.log_metric("cv_min", float(cv_scores.min()))
                mlflow.log_metric("cv_max", float(cv_scores.max()))
            except Exception as e:
                print(f"Warning: Could not calculate CV scores: {e}")
            
            # Create and log artifacts
            print("🔄 Creating visualizations...")
            
            # Feature importance plot
            fi_path = create_feature_importance_plot(model, feature_names, args.model_name)
            if fi_path and os.path.exists(fi_path):
                try:
                    mlflow.log_artifact(fi_path)
                    print("✅ Feature importance plot logged to MLflow")
                except Exception as e:
                    print(f"⚠️  Warning: Could not log feature importance to MLflow: {e}")
            
            # Confusion matrix
            cm_path = create_confusion_matrix_plot(y_test, y_pred, args.model_name)
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
                report_path = f'artifacts/classification_report_{args.model_name}.txt'
                with open(report_path, 'w') as f:
                    f.write(report)
                mlflow.log_artifact(report_path)
                print("✅ Classification report logged to MLflow")
            except Exception as e:
                print(f"Warning: Could not create/log classification report: {e}")
            
            # Log model with multiple fallback strategies
            print("🔄 Logging model to MLflow...")
            model_logged, model_uri = log_model_safely(model, X_train, args.model_name, use_dagshub)
            
            if not model_logged:
                print("❌ All model logging strategies failed!")
                # Save locally as final fallback
                local_path = save_model_locally(model, f"final_fallback_{args.model_name}")
                if local_path:
                    model_uri = local_path
            
            # Log model URI for Docker building
            if model_uri:
                mlflow.log_param("model_uri", model_uri)
                # Save model URI to file for GitHub Actions
                with open("model_uri.txt", "w") as f:
                    f.write(model_uri)
                print(f"✅ Model URI saved to file: {model_uri}")
            
            # Print summary
            print("\n" + "="*60)
            print("🎉 CI TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Model parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
            print(f"Test accuracy: {metrics['accuracy']:.4f}")
            print(f"MLflow Run ID: {run.info.run_id}")
            if model_uri:
                print(f"Model URI: {model_uri}")
            
            # Check if we can get experiment info
            try:
                experiment = mlflow.get_experiment_by_name(args.experiment_name)
                if experiment:
                    print(f"Experiment ID: {experiment.experiment_id}")
                    if use_dagshub:
                        print(f"DagsHub MLflow URL: https://dagshub.com/wildanmr/SMSML_Wildan-Mufid-Ramadhan.mlflow/#/experiments/{experiment.experiment_id}/runs/{run.info.run_id}")
                    else:
                        print(f"Local MLflow UI: http://localhost:5000")
            except Exception as e:
                print(f"Could not get experiment info: {e}")
            
            print("="*60)
            
            return model, metrics, run.info.run_id, model_uri
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting CI MLflow Training...")
    print("="*60)
    
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"📋 Training Configuration:")
    print(f"   Data Path: {args.data_path}")
    print(f"   Use DagsHub: {args.use_dagshub}")
    print(f"   Experiment: {args.experiment_name}")
    print(f"   Model Name: {args.model_name}")
    print(f"   Parameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print("="*60)
    
    try:
        model, metrics, run_id, model_uri = train_model_with_parameters(args)
        print("\n✅ CI Training completed successfully!")
        
        if args.use_dagshub.lower() == 'true':
            print("🔗 Check your DagsHub repository for MLflow tracking results:")
            print("   https://dagshub.com/wildanmr/SMSML_Wildan-Mufid-Ramadhan.mlflow")
        else:
            print("🔗 Check your local MLflow UI:")
            print("   Run 'mlflow ui' in terminal and go to http://localhost:5000")
        
        # Exit with success code
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ CI Training failed: {e}")
        print("Please check your data path, column names, and authentication.")
        sys.exit(1)