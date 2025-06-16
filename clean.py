import shutil
import os

hf_cache = os.path.expanduser("~/.cache/huggingface")
shutil.rmtree(hf_cache, ignore_errors=True)
print("Hugging Face cache cleared.")
