import os
from huggingface_hub import snapshot_download, HfApi

class HuggingFaceWrapper:
    def __init__(self, hugging_face_api_key):
        self.hugging_face_api_key = hugging_face_api_key
        self.api = HfApi()

    @staticmethod
    def get_model(repo_id, local_dir, hugging_face_api_key):           
        # Download repo from HuggingFace
        print(f"Checking if {repo_id} already exists in {local_dir}")
        if os.path.exists(local_dir):
            print(f"{repo_id} already exists in {local_dir}")
            return local_dir
        
        print(f"{local_dir} does not exist, creating it")
        os.makedirs(local_dir, exist_ok=True)
        print(f"Downloading {repo_id} to {local_dir}")
        local_dir = snapshot_download(
            repo_id=repo_id, 
            local_dir=local_dir, 
            repo_type="model", 
            token=hugging_face_api_key
        )
        
        print(f"{repo_id} downloaded to {local_dir}")

        return local_dir

    def upload_model(self, repo_id, local_dir, commit_message="Update model"):
        """
        Upload a model to Hugging Face.
        
        :param repo_id: The ID of the repository (e.g., 'username/repo-name')
        :param local_dir: The local directory containing the model files
        :param commit_message: The commit message for this update
        """
        try:
            # Ensure the repository exists (create if it doesn't)
            try:
                self.api.create_repo(repo_id=repo_id, token=self.hugging_face_api_key, exist_ok=True)
            except Exception as e:
                print(f"Note: {e}")

            # Upload all files in the directory
            for root, _, files in os.walk(local_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    repo_path = os.path.relpath(file_path, local_dir)
                    
                    self.api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        token=self.hugging_face_api_key
                    )
                    print(f"Uploaded {file} to {repo_id}")

            # Create a commit with all the uploaded files
            self.api.create_commit(
                repo_id=repo_id,
                operations="push",
                commit_message=commit_message,
                token=self.hugging_face_api_key
            )
            
            print(f"Successfully uploaded and committed model to {repo_id}")
        except Exception as e:
            print(f"An error occurred while uploading the model: {e}")