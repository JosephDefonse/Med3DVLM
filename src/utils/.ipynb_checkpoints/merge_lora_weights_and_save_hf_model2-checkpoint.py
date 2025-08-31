# merge_clean.py
import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import transformers
from transformers import AutoTokenizer

from src.model.llm.qwen import VLMQwenForCausalLM

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-7B-Instruct")
    model_type: Optional[str] = field(default="vlm_qwen")

    # vision/projector must mirror training
    vision_tower: Optional[str] = field(default="dcformer")
    vision_select_layer: Optional[int] = field(default=-2)
    vision_select_feature: Optional[str] = field(default="cls_patch")
    pretrain_vision_model: Optional[str] = field(default=None)
    pretrain_clip_model: Optional[str] = field(default=None)
    freeze_vision_tower: bool = field(default=False)

    # projector config (mirror training)

    freeze_backbone: bool = field(default=False)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(
        default=False,
        metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"},
    )
    pretrain_mm_mlp_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained mm_projector and embed_tokens."},
    )
    
    mm_projector_type: Optional[str] = field(default="mixer")
    mm_mlp_depth: int = field(default=2)
    proj_layer_type: str = field(default="mlp")
    proj_layer_num: int = field(default=2)
    proj_pooling_type: str = field(default="spatial")
    proj_pooling_size: int = field(default=2)
    proj_residual: bool = field(default=False)
    low_output_size: List[int] = field(default_factory=lambda: [192, 128])
    high_output_size: List[int] = field(default_factory=lambda: [64, 128])

    # tokenizer/image
    input_size: tuple = field(default=(256, 256, 128))
    patch_size: tuple = field(default=(16, 16, 16))
    dim: int = field(default=768)
    depth: int = field(default=12)

    # paths
    adapter_dir: str = field(default="./output/Med3DVLM-Qwen-2.5-7B-finetune_vqa_17")
    non_lora_path: Optional[str] = field(default=None)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./models/Med3DVLM-Qwen-2.5-7B-vqa-17")

def _proj_out_num(mm_projector_type: str) -> int:
    if mm_projector_type == "low_high_mlp":
        return 288
    if mm_projector_type in ("mlp", "mhsa"):
        return 32
    return 256  # mixer

def _normalize_nonlora_keys(sd: dict) -> dict:
    """Strip PEFT prefixes so keys match the base VLM."""
    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("base_model."):
            nk = nk[len("base_model."):]
        # collapse model.model. -> model.
        nk = nk.replace("model.model.", "model.")
        out[nk] = v
    return out

def _find_adapter_file(adapter_dir: str) -> str:
    safetensors = os.path.join(adapter_dir, "adapter_model.safetensors")
    binf = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(safetensors):
        return safetensors
    if os.path.exists(binf):
        return binf
    raise FileNotFoundError(
        f"No adapter_model.safetensors or adapter_model.bin in {adapter_dir}. "
        "Make sure you saved with get_peft_model_state_dict()."
    )

def main():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    margs, targs = parser.parse_args_into_dataclasses()

    adapter_dir = margs.adapter_dir
    non_lora_path = margs.non_lora_path or os.path.join(adapter_dir, "non_lora_weights.bin")
    os.makedirs(targs.output_dir, exist_ok=True)

    print("==> Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        margs.model_name_or_path,
        cache_dir=targs.cache_dir,
        padding_side="right",
        use_fast=False
    )
    
    tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>"]})
    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    margs.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    print("vocab_size:", len(tokenizer))

    print("==> Load base VLM")
    model = VLMQwenForCausalLM.from_pretrained(
        margs.model_name_or_path,
        cache_dir=targs.cache_dir,
        torch_dtype="auto"
    )

    # init vision/projector same as training
    margs.proj_out_num = _proj_out_num(margs.mm_projector_type)
    margs.num_new_tokens = 1
    model.get_model().initialize_vision_modules(model_args=margs)
    model.initialize_vision_tokenizer(margs, tokenizer)

    # 1) load NON-LoRA weights (normalize keys first)
    if os.path.exists(non_lora_path):
        print(f"==> Load non-LoRA weights: {non_lora_path}")
        non_lora = torch.load(non_lora_path, map_location="cpu")
        non_lora = _normalize_nonlora_keys(non_lora)
        missing, unexpected = model.load_state_dict(non_lora, strict=False)
        print(f"non-LoRA load -> missing: {len(missing)}, unexpected: {len(unexpected)}")
        if unexpected[:10]:
            print("  unexpected (first 10):", unexpected[:10])
    else:
        print(f"WARNING: non_lora_weights.bin not found at {non_lora_path}; "
              "fine-tuned projector/vision will NOT be applied.")

    # 2) force-load adapter file only
    adapter_file = _find_adapter_file(adapter_dir)
    print(f"==> Load PEFT adapters from: {adapter_file}")
    from peft import PeftModel, PeftConfig
    peft_cfg = PeftConfig.from_pretrained(adapter_dir)
    model = PeftModel(model, peft_cfg)

    # -------------------- DEBUG ---------------------------------------
    # 1) Inspect what PEFT expects (the wrapped model's LoRA keys)
    msd = model.state_dict()
    expected_lora = [k for k in msd.keys() if "lora_" in k]
    print("expected lora keys:", len(expected_lora))
    print("sample expected:", expected_lora[:5])
    # ----------------------------------------------------------------
    
    # load the adapter weights explicitly
    if adapter_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        adapter_sd = load_file(adapter_file)
    else:
        adapter_sd = torch.load(adapter_file, map_location="cpu")

    load_result = model.load_state_dict(adapter_sd, strict=False)

    # -------------------- DEBUG ---------------------------------------
    
    bad = [(k, tuple(v.shape)) for k, v in adapter_sd.items() if any(d == 0 for d in v.shape)]
    print("zero-sized adapter tensors:", len(bad))
    print("sample:", bad[:5])

    # 2) Inspect what's in the adapter file
    adapter_lora = [k for k in adapter_sd.keys() if "lora_" in k]
    print("adapter lora keys:", len(adapter_lora))
    print("sample adapter:", adapter_lora[:5])
    
    # 3) Show the first few 'unexpected' and whether any missing are actually LoRA
    missing = getattr(load_result, "missing_keys", load_result[0])
    unexpected = getattr(load_result, "unexpected_keys", load_result[1])
    
    missing_lora = [k for k in missing if "lora_" in k]
    print("LoRA load -> missing:", len(missing), "unexpected:", len(unexpected))
    print("missing_lora (first 5):", missing_lora[:5])
    print("unexpected (first 5):", unexpected[:5])
    
    missing = getattr(load_result, "missing_keys", load_result[0])
    unexpected = getattr(load_result, "unexpected_keys", load_result[1])
    
    print(f"LoRA load -> missing: {len(missing)}, unexpected: {len(unexpected)}")
    
    # sanity: keys should be LoRA keys; no embed_tokens/lm_head full weights
    bad = [k for k in adapter_sd.keys() if "lora_" not in k]
    if bad:
        print("WARNING: adapter file contains non-LoRA keys (first 10):", bad[:10])
    # ----------------------------------------------------------------
    
    print("==> Merging LoRA into base (merge_and_unload)")
    model = model.merge_and_unload()

    # After tokenizer creation + any add_special_tokens + model init
    tok_len = len(tokenizer)
    emb_rows = model.get_input_embeddings().weight.size(0)
    
    # If the model's embedding matrix doesn't match the tokenizer, resize the *model*
    if emb_rows != tok_len:
        print(f"Resizing token embeddings: {emb_rows} -> {tok_len}")
        model.resize_token_embeddings(tok_len)

    # NEW: lock the vocab in config so the class won't add more tokens on load
    model.config.vocab_size = tok_len
    if hasattr(model.config, "num_new_tokens"):
        model.config.num_new_tokens = 0  # <— IMPORTANT
    
    # 3) save final
    print(f"==> Save merged model to: {targs.output_dir}")
    model.config.architectures = [type(model).__name__]
    model._name_or_path = targs.output_dir
    model.config.save_pretrained(targs.output_dir)
    model.save_pretrained(targs.output_dir)
    tokenizer.save_pretrained(targs.output_dir)

    # optional: keep a copy of the vision tower
    vision_sd = model.get_model().vision_tower.state_dict()
    torch.save(vision_sd, os.path.join(targs.output_dir, "vision_tower.bin"))

    print("✅ Done.")

if __name__ == "__main__":
    main()
