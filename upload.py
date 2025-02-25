from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="head_model.h5",  # Your model file
    path_in_repo="head_model.h5",  # Name in the repo
    repo_id="srikanth-maganti/Retinitis-pigmentosa-detection",  # Your Hugging Face repo
    repo_type="model"  # Define it as a model repo
)