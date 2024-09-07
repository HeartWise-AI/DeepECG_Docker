


import os

from huggingface_hub import snapshot_download


class HuggingFaceWrapper:
    @staticmethod
    def get_model(model_name, repo_id, local_dir, hugging_face_api_key):           
        # Download repo from HuggingFace
        os.makedirs(local_dir, exist_ok=True)
        local_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type="model", token=hugging_face_api_key)
        
        print(f"{model_name} downloaded to {local_dir}")

        return local_dir
