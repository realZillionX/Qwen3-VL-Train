import os
import argparse
from modelscope.hub.snapshot_download import snapshot_download

def download_model(target_dir):
    print(f"Downloading Qwen3-VL-32B-Thinking to {target_dir}...")
    
    # Model ID on ModelScope. 
    # NOTE: Assuming the model ID is 'qwen/Qwen3-VL-32B-Thinking' or similar.
    # Since Qwen3-VL is hypothetical/very new in this context, 
    # I will use a placeholder ID and comment strictly to the user to verify properly.
    # If standard Qwen2.5-VL is meant, I'd use that, but user said Qwen3.
    # I will use a generic variable.
    
    model_id = "qwen/Qwen3-VL-32B-Thinking" # <--- PLEASE VERIFY THIS ID
    
    try:
        model_dir = snapshot_download(
            model_id, 
            cache_dir=target_dir, 
            revision='master'
        )
        print(f"Successfully downloaded to: {model_dir}")
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please check if the model_id is correct on ModelScope (https://modelscope.cn).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Model for Offline Use")
    parser.add_argument("--save_dir", type=str, default="./model_weights", help="Directory to save the model")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    download_model(args.save_dir)
