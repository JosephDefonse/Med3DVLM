# rebuild_adapter.py
import torch
from transformers import AutoTokenizer
from src.model.llm.qwen import VLMQwenForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-7B-Instruct"
BROKEN = "./output/Med3DVLM-Qwen-2.5-7B-lora-adapter-broken"
CLEAN  = "./output/Med3DVLM-Qwen-2.5-7B-lora-adapter"

def main():
    # 1) load the base model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=False)
    # (re‑add your special vision token exactly as in training)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>"]})

    model = VLMQwenForCausalLM.from_pretrained(BASE)
    # resize the embedding to match tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # 2) wrap it in PEFT
    peft = PeftModel.from_pretrained(model, BROKEN, torch_dtype=torch.float32)

    # 3) now re‑export _just_ the adapter
    peft.save_pretrained(CLEAN)
    tokenizer.save_pretrained(CLEAN)

    print("✅ rebuilt adapter into", CLEAN)

if __name__ == "__main__":
    main()
