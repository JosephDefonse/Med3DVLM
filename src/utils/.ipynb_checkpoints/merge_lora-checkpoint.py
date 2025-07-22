# merge_lora.py
import argparse
from src.model.llm.qwen import VLMQwenForCausalLM
from peft import PeftModel

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_dir", required=True)
    p.add_argument("--out_dir",   required=True)
    args = p.parse_args()

    # 1) load the base
    base = VLMQwenForCausalLM.from_pretrained(args.base_model)

    # 2) load your LoRA adapters
    peft = PeftModel.from_pretrained(base, args.adapter_dir)

    # 3) merge + unload the PEFT wrapper
    merged = peft.merge_and_unload()

    # 4) save the fully merged model
    merged.save_pretrained(args.out_dir)
    print("✅ merged model saved to", args.out_dir)

if __name__=="__main__":
    main()
