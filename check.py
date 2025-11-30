from dotenv import load_dotenv
import os

load_dotenv()

print("HF_REPO_ID:", os.getenv("HF_REPO_ID"))
print("HF_SUBFOLDER:", os.getenv("HF_SUBFOLDER"))
print("KERAS_MODEL_PATH:", os.getenv("KERAS_MODEL_PATH"))
