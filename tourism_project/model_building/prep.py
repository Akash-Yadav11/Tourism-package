
# for data manipulation
import pandas as pd
import sklearn
import os

# for splitting data
from sklearn.model_selection import train_test_split

# for huggingface upload
from huggingface_hub import HfApi

# API setup
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset
DATASET_PATH = "hf://datasets/akashyadav2005/tourism_project/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unnecessary columns
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)


# DUMMY ENCODING (ONE-HOT)
categorical_cols = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Target variable
target_col = 'ProdTaken'

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload to HuggingFace
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="akashyadav2005/tourism_project",
        repo_type="dataset",
    )
