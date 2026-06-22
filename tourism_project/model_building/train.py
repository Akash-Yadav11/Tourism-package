# Step 1: Write the correct train.py file
%%writefile tourism_project/model_building/train.py
import pandas as pd
import os
import joblib
import xgboost as xgb
import mlflow

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

print("=" * 60)
print("TOURISM PACKAGE MODEL TRAINING")
print("=" * 60)

# Load dataset from Hugging Face
print("\n📥 Loading dataset from Hugging Face...")
DATA_PATH = "hf://datasets/akashyadav2005/tourism_project/tourism.csv"
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Clean
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True, errors='ignore')

target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

# Split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Train: {Xtrain.shape[0]} samples, Test: {Xtest.shape[0]} samples")

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

print(f"\n📊 Features:")
print(f"  Numeric: {len(numeric_features)}")
print(f"  Categorical: {len(categorical_features)}")

# Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Model
model = xgb.XGBClassifier(random_state=42)

# Pipeline
pipeline = make_pipeline(preprocessor, model)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.01, 0.1]
}

# Set MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

with mlflow.start_run() as run:
    print(f"\n🔬 MLflow Run ID: {run.info.run_id}")
    
    print("\n🔄 Running GridSearchCV...")
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(Xtrain, ytrain)

    best_model = grid.best_estimator_
    print(f"\n✅ Best parameters: {grid.best_params_}")

    preds = best_model.predict(Xtest)
    report = classification_report(ytest, preds, output_dict=True)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics({
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"]
    })

    # Save model
    model_path = "best_tourism_project_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    
    print(f"\n📊 Results:")
    print(f"  Test Accuracy: {report['accuracy']:.4f}")
    print(f"  Test Precision: {report['1']['precision']:.4f}")
    print(f"  Test Recall: {report['1']['recall']:.4f}")
    print(f"  Test F1: {report['1']['f1-score']:.4f}")
    print(f"\n✅ Model saved to {model_path}")

    # Upload to Hugging Face
    try:
        from huggingface_hub import HfApi, create_repo
        from huggingface_hub.utils import RepositoryNotFoundError
        
        api = HfApi(token=os.getenv("HF_TOKEN"))
        repo_id = "akashyadav2005/tourism_project_model"
        repo_type = "model"
        
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            print(f"\n✅ Repository '{repo_id}' exists.")
        except RepositoryNotFoundError:
            print(f"\nCreating repository '{repo_id}'...")
            create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=model_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✅ Model uploaded to: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"⚠️ Upload failed: {e}")

