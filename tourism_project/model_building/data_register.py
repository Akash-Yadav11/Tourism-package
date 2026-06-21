from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os


api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "akashyadav2005/tourism_project"
repo_type = "dataset"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
