import os
# Force offline mode
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['MS_OFFLINE'] = '1'

import torch
from swift.llm import get_model_tokenizer, get_template, get_dataset, sft_main, SftArguments
from swift.utils import get_logger

logger = get_logger()

def main():
    # User args
    # You can also use CLI arguments by calling `sft_main()` without arguments and passing via command line,
    # but here we wrap it for consistency with the GRPO script style.
    
    # Placeholder path - User will need to update this
    model_path = "/path/to/Qwen3-VL-32B-Thinking" 
    data_path = "train.jsonl"
    output_dir = "output/sft_qwen3_vl"
    
    # Swift SFT Arguments
    # Ref: https://github.com/modelscope/ms-swift/blob/main/docs/source/LLM/SFT%E5%8F%82%E6%95%B0.md
    sft_args = SftArguments(
        model=model_path,
        model_type='qwen2-vl-7b-instruct', # or appropriate mapping if known, else auto-detect from model config usually works if model path is valid
        # If loading local model, usually 'model' arg is enough.
        
        dataset=[data_path], 
        # ms-swift handles jsonl dataset loading if we pass just the path.
        # It expects 'query', 'response', 'images'.
        
        output_dir=output_dir,
        
        # Training Config
        learning_rate=2e-5,
        num_train_epochs=1,
        per_device_train_batch_size=1, # Adjust based on VRAM
        gradient_accumulation_steps=16, # Increase for effective batch size
        save_steps=100,
        logging_steps=10,
        max_length=2048,
        
        # Precision
        bf16=True,
        
        # LoRA Config
        sft_type='lora',
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_rank=16,
        lora_alpha=32,
        
        # Multimodal
        # Qwen-VL handles images automatically via transformers/swift template
        
        # System control
        report_to=['tensorboard'],
        use_vllm=False, # SFT usually uses standard forward pass, not vLLM generation
    )
    
    print("Starting SFT Training...")
    result = sft_main(sft_args)
    print(f"SFT finished. Output in {output_dir}")

if __name__ == "__main__":
    main()
