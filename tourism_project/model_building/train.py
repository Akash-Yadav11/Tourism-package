import pandas as pd
import os
import joblib
import xgboost as xgb
import mlflow
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

print("=" * 60)
print("TOURISM PACKAGE MODEL TRAINING (FIXED FOR IMBALANCE)")
print("=" * 60)

# Load dataset from Hugging Face
print("\n Loading dataset from Hugging Face...")
DATA_PATH = "hf://datasets/akashyadav2005/tourism_project/tourism.csv"
df = pd.read_csv(DATA_PATH)
print(f" Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Clean
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True, errors='ignore')

target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

# Check class distribution
print(f"\n Class Distribution:")
print(y.value_counts())
print(f"Positive class (1) percentage: {y.mean() * 100:.2f}%")

# Split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f" Train: {Xtrain.shape[0]} samples, Test: {Xtest.shape[0]} samples")

# Features
numeric_features = [
    'Age','CityTier','DurationOfPitch','NumberOfPersonVisiting',
    'NumberOfFollowups','PreferredPropertyStar','NumberOfTrips',
    'Passport','PitchSatisfactionScore','OwnCar',
    'NumberOfChildrenVisiting','MonthlyIncome'
]

categorical_features = [
    'TypeofContact','Occupation','Gender',
    'ProductPitched','MaritalStatus','Designation'
]

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
)

# Model with class weight
class_weight = len(ytrain[ytrain == 0]) / len(ytrain[ytrain == 1])
print(f"\n Class weight: {class_weight:.2f}")

model = xgb.XGBClassifier(
    random_state=42,
    scale_pos_weight=class_weight,  # This handles class imbalance
    eval_metric='logloss',
    use_label_encoder=False
)

# Pipeline
pipeline = make_pipeline(preprocessor, model)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [4, 6, 8],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
    "xgbclassifier__min_child_weight": [1, 3, 5]
}

# Set MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

print("\n" + "=" * 60)
print("STARTING TRAINING WITH IMBALANCE HANDLING")
print("=" * 60)

with mlflow.start_run() as run:
    print(f"\n MLflow Run ID: {run.info.run_id}")
    
    print("\n Running GridSearchCV...")
    grid = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        n_jobs=-1, 
        verbose=1,
        scoring='roc_auc'  # Better metric for imbalanced data
    )
    grid.fit(Xtrain, ytrain)

    best_model = grid.best_estimator_
    print(f"\n Best parameters: {grid.best_params_}")
    mlflow.log_params(grid.best_params_)

    # Get predictions with different thresholds
    y_pred_proba = best_model.predict_proba(Xtest)[:, 1]
    
    # Try different thresholds to find optimal
    thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]
    best_threshold = 0.5
    best_f1 = 0
    
    print("\n Finding optimal threshold:")
    for thresh in thresholds:
        preds = (y_pred_proba >= thresh).astype(int)
        report = classification_report(ytest, preds, output_dict=True)
        f1 = report.get('1', {}).get('f1-score', 0)
        recall = report.get('1', {}).get('recall', 0)
        precision = report.get('1', {}).get('precision', 0)
        print(f"  Threshold {thresh:.2f}: F1={f1:.3f}, Recall={recall:.3f}, Precision={precision:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    print(f"\n Best threshold: {best_threshold:.2f} (F1={best_f1:.3f})")
    mlflow.log_metric("best_threshold", best_threshold)
    
    # Use best threshold
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # Generate classification report
    report = classification_report(ytest, y_pred, output_dict=True)
    cm = confusion_matrix(ytest, y_pred)
    auc_roc = roc_auc_score(ytest, y_pred_proba)
    
    # Log metrics
    mlflow.log_metrics({
        "accuracy": report["accuracy"],
        "precision_class1": report.get('1', {}).get('precision', 0),
        "recall_class1": report.get('1', {}).get('recall', 0),
        "f1_class1": report.get('1', {}).get('f1-score', 0),
        "auc_roc": auc_roc,
        "best_threshold": best_threshold
    })
    
    # Save model
    model_path = "best_tourism_project_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    
    # Save threshold for later use
    with open('threshold.txt', 'w') as f:
        f.write(str(best_threshold))
    mlflow.log_artifact('threshold.txt')
    
    print(f"\n Results:")
    print(f"  Confusion Matrix:")
    print(f"    TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"    FN: {cm[1][0]}, TP: {cm[1][1]}")
    print(f"  Accuracy: {report['accuracy']:.4f}")
    print(f"  Precision (Class 1): {report.get('1', {}).get('precision', 0):.4f}")
    print(f"  Recall (Class 1): {report.get('1', {}).get('recall', 0):.4f}")
    print(f"  F1 (Class 1): {report.get('1', {}).get('f1-score', 0):.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"\n Model saved to {model_path}")
    
    # Upload to Hugging Face
    try:
        from huggingface_hub import HfApi, create_repo
        from huggingface_hub.utils import RepositoryNotFoundError
        
        api = HfApi(token=os.getenv("HF_TOKEN"))
        repo_id = "akashyadav2005/tourism_project_model"
        repo_type = "model"
        
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            print(f"\n Repository '{repo_id}' exists.")
        except RepositoryNotFoundError:
            print(f"\nCreating repository '{repo_id}'...")
            create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        
        # Upload model
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=model_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        
        # Upload threshold
        api.upload_file(
            path_or_fileobj='threshold.txt',
            path_in_repo='threshold.txt',
            repo_id=repo_id,
            repo_type=repo_type,
        )
        
        print(f" Model uploaded to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f" Upload failed: {e}")

