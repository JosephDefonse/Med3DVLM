import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import json

import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, EarlyStoppingCallback

import wandb
from src.dataset.mllm_dataset import TextDatasets, TextYNDatasets, QADatasets
from src.model.llm.qwen import VLMQwenForCausalLM
from src.train.trainer import MLLMTrainer

from peft import get_peft_model_state_dict
from safetensors.torch import save_file

import torch.distributed as dist

import re

tokenizer = None

def is_rank_zero():
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            return False
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return False
    return True


def rank0_print(*args):
    if is_rank_zero():
        print(*args)


@dataclass
class ModelArguments:
    wb_name: Optional[str] = field(default="MLLM")
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "Path to the LLM or MLLM."},
    )
    model_type: Optional[str] = field(default="vlm_qwen")

    freeze_backbone: bool = field(default=True)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(
        default=True,
        metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"},
    )
    pretrain_mm_mlp_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained mm_projector and embed_tokens."},
    )

    # image
    input_size: tuple = field(default=(256, 256, 128))
    patch_size: int = field(default=(16, 16, 16))
    dim: int = field(default=768)
    depth: int = field(default=12)

    # vision
    vision_tower: Optional[str] = field(default="dcformer")
    vision_select_layer: Optional[int] = field(default=-2)
    vision_select_feature: Optional[str] = field(default="cls_patch")
    pretrain_vision_model: str = field(
        default=None, metadata={"help": "Path to pretrained model for ViT."}
    )
    pretrain_clip_model: str = field(
        default=None, metadata={"help": "Path to pretrained model for CLIP."}
    )
    freeze_vision_tower: bool = field(default=False)

    # projector
    mm_projector_type: Optional[str] = field(default="mlp")
    mm_mlp_depth: int = field(
        default=2, metadata={"help": "Depth of MLP in projector."}
    )

    low_output_size: List[int] = field(
        default_factory=lambda: [192, 128],
        metadata={"help": "Output size of low feature."},
    )
    high_output_size: List[int] = field(
        default_factory=lambda: [64, 128],
        metadata={"help": "Output size of high feature."},
    )


@dataclass
class DataArguments:
    data_root: str = field(
        default="./data/", metadata={"help": "Root directory for all data."}
    )

    # caption data
    cap_data_path: str = field(
        default="./data/M3D_Cap_npy/M3D_Cap.json",
        metadata={"help": "Path to caption data."},
    )

    # VQA data
    vqa_data_train_path: str = field(
        default="./data/M3D-VQA/M3D_VQA_train.csv",
        metadata={"help": "Path to training VQA data."},
    )
    vqa_data_val_path: str = field(
        default="./data/M3D-VQA/M3D_VQA_val.csv",
        metadata={"help": "Path to validation VQA data."},
    )
    vqa_data_test_path: str = field(
        default="./data/M3D-VQA/M3D_VQA_test.csv",
        metadata={"help": "Path to testing VQA data."},
    )

    vqa_yn_data_train_path: str = field(
        default="./data/M3D-VQA/M3D_VQA_yn_train.csv",
        metadata={"help": "Path to training VQA Yes or No data."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # lora
    lora_enable: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    lora_weight_path: str = ""
    lora_bias: str = "none"

    cache_dir: Optional[str] = field(default=None)
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=1024,  # 1024
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    seed: int = 42
    ddp_backend: str = "nccl"
    ddp_timeout: int = 128000
    ddp_find_unused_parameters: bool = True
    optim: str = field(default="adamw_torch")

    # This is set up to facilitate debugging, pls config these in bash file in training.
    bf16: bool = True
    output_dir: str = "./output/Med3DVLM-pretrain-test"
    num_train_epochs: float = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    eval_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 2
    learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 10  # 0.001
    gradient_checkpointing: bool = False  # train fast
    dataloader_pin_memory: bool = True  # fast
    dataloader_num_workers: int = 0
    report_to: str = "tensorboard"
    
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = True
    load_best_model_at_end: bool = True
    
    max_grad_norm: float = 0.5
    label_smoothing_factor: float = 0.05

# def compute_metrics(eval_preds):
#     labels_ids = eval_preds.label_ids
#     pred_ids = eval_preds.predictions

#     labels = labels_ids[:, 1:]
#     preds = pred_ids[:, :-1]

#     labels_flatten = labels.reshape(-1)
#     preds_flatten = preds.reshape(-1)
#     valid_indices = np.where(labels_flatten != -100)
#     filtered_preds = preds_flatten[valid_indices]
#     filtered_labels = labels_flatten[valid_indices]
#     acc_score = sum(filtered_preds == filtered_labels) / len(filtered_labels)

#     return {"accuracy": acc_score}

DEBUG_METRICS_N = 6

def compute_metrics(eval_preds):
    pred_ids   = eval_preds.predictions   # from preprocess_logits_for_metrics (argmax ids)
    labels_ids = eval_preds.label_ids

    # shift like LM loss
    preds  = pred_ids[:, :-1]
    labels = labels_ids[:, 1:]

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    def trim_ids(ids):
        # accept list/int/np array and make it at least 1D
        arr = np.atleast_1d(np.array(ids, dtype=np.int64))
        if arr.size == 0:
            return arr
        if eos_id is not None:
            eos_idx = np.nonzero(arr == eos_id)[0]
            if eos_idx.size:
                arr = arr[:eos_idx[0]]
        if pad_id is not None:
            arr = arr[arr != pad_id]
        arr = arr[arr != -100]
        return arr

    def safe_decode(arr):
        arr = trim_ids(arr)
        if arr.size == 0:
            return ""
        return tokenizer.decode(arr.tolist(), skip_special_tokens=True)

    def norm_text(s): return re.sub(r"\s+", " ", s).strip().lower()
    def extract_letter(s):
        m = re.match(r"^\s*([A-La-l])\s*[\.\)]?", s)
        return m.group(1).upper() if m else None

    n = labels.shape[0]
    token_hits = token_total = 0
    string_hits = choice_hits = 0

    # limited debug
    already = getattr(compute_metrics, "_printed", 0)
    todo = max(0, min(DEBUG_METRICS_N - already, n))
    if todo > 0:
        print("\n=== compute_metrics DEBUG ===")

    for i in range(n):
        mask = labels[i] != -100
        if not np.any(mask):
            continue

        ref_ids = labels[i][mask]
        hyp_ids = preds[i][mask]

        # token-level acc on the masked window
        L = min(ref_ids.shape[0], hyp_ids.shape[0])
        if L > 0:
            token_hits  += int((hyp_ids[:L] == ref_ids[:L]).sum())
            token_total += L

        ref_text = safe_decode(ref_ids)
        hyp_text = safe_decode(hyp_ids)

        if norm_text(hyp_text) == norm_text(ref_text):
            string_hits += 1

        ref_letter = extract_letter(ref_text)
        hyp_letter = extract_letter(hyp_text)
        if ref_letter is not None and hyp_letter == ref_letter:
            choice_hits += 1

        if todo > 0:
            print(f"[{i}] REF: {ref_text}")
            print(f"[{i}] HYP: {hyp_text}")
            print(f"[{i}] ref_letter={ref_letter}  hyp_letter={hyp_letter}")
            print("---")
            todo -= 1

    if DEBUG_METRICS_N:
        compute_metrics._printed = already + (DEBUG_METRICS_N - max(0, todo))

    return {
        "token_acc":  token_hits / max(1, token_total),
        "string_acc": string_hits / max(1, n),
        "choice_acc": choice_hits / max(1, n),
    }
    
def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save projector and embed_tokens in pretrain
        keys_to_match = ["mm_projector", "embed_tokens", "embeddings"]

        weight_to_save = get_mm_projector_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = [
        "vision_tower",
        "mm_projector",
        "embed_tokens",
        "lm_head",
    ]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

@dataclass
class DataCollator:
    def __call__(self, batch: list) -> dict:
        images, input_ids, labels, attention_mask = tuple(
            [b[key] for b in batch]
            for key in ("image", "input_id", "label", "attention_mask")
        )

        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

        return_dict = dict(
            images=images,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        return return_dict


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank0_print("=" * 20 + " Tokenizer preparation " + "=" * 20)
    
    # Load tokenizer from the given path with specified configurations
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    print(tokenizer)

    def snap(tag, tok):
        base = tok.vocab_size
        added_map = tok.get_added_vocab()  # dict[str->id] for added tokens
        total = len(tok)                   # base + added
        print(f"[{tag}] base={base} added_count={len(added_map)} total={total}")
        # show a few added tokens with ids
        for k in list(added_map.keys())[:]:
            print(f"  added: {k!r} -> {added_map[k]}")
        print("all_special_tokens:", tok.all_special_tokens)
        print("eos_id:", tok.eos_token_id, "pad_id:", tok.pad_token_id)
    
    # snap("before", tokenizer)
        
    # Define and add special tokens
    # special_token = {"additional_special_tokens": ["<im_patch>"]}
    # tokenizer.add_special_tokens(special_token)
    extras = list(dict.fromkeys((tokenizer.additional_special_tokens or []) + ["<im_patch>"]))
    tokenizer.add_special_tokens({"additional_special_tokens": extras})

    # print("---------------------------------------------------------------------")
    # print(tokenizer)
    
    # It doesn't look like unk_token exists (is None), hence it won't go through this
    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # type is vlm_qwen, hence it won't go through this
    if "llama3" in model_args.model_type:
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token

    """
    INITIALISING MODEL ARGUMENTS FROM TOKENIZER FOR MODEL PREPARATION
    """

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    # print(model_args.img_token_id) # this represents the numerical id of <img_patch>
    
    model_args.vocab_size = len(tokenizer)
    # print(model_args.vocab_size)
    rank0_print("vocab_size: ", model_args.vocab_size)
    # snap("after", tokenizer)
    
    # mm_projector_type = mixer, hence, = 256
    if model_args.mm_projector_type is not None:
        if model_args.mm_projector_type == "low_high_mlp":
            model_args.proj_out_num = 288
        elif (
            model_args.mm_projector_type == "mlp"
            or model_args.mm_projector_type == "mhsa"
        ):
            model_args.proj_out_num = 32
        else:
            model_args.proj_out_num = 256
    else:
        raise ValueError(f"Unknown Projector Type {model_args.mm_projector_type}")

    rank0_print("=" * 20 + " Model preparation " + "=" * 20)
    if model_args.vision_tower is not None:
        if "qwen" in model_args.model_type:
            model = VLMQwenForCausalLM.from_pretrained(
                model_args.model_name_or_path, cache_dir=training_args.cache_dir
            )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir
        )
    
    
    model.config.use_cache = False

    if model_args.freeze_backbone: # SET TO TRUE
        model.model.requires_grad_(False)

    model.enable_input_require_grads()
    
    if training_args.gradient_checkpointing: # SET TO TRUE IN SH SCRIPT
        model.gradient_checkpointing_enable()

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
        model_args.tune_mm_mlp_adapter
    )
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    model_args.num_new_tokens = 1
    model.initialize_vision_tokenizer(model_args, tokenizer)

    # IGNORE - NOT USING
    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("load pretrained MLLM weights.")

    # USING FOR FINE_TUNING
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)
        
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in [
                        "vision_tower",
                        "mm_projector",
                        # "embed_tokens",
                        # "lm_head",
                    ]
                ]
            ):
                p.requires_grad = True

        model.print_trainable_parameters()


    # def hook_shape(name):
    #     def _h(mod, inp, out):
    #         def _shape(o):
    #             if isinstance(o, torch.Tensor): return tuple(o.shape)
    #             if isinstance(o, (list, tuple)): return [tuple(t.shape) for t in o]
    #             return type(o).__name__
    #         print(f"[hook] {name}: { _shape(out) }")
    #     return _h
    
    # vt   = model.get_model().get_vision_tower()
    # proj = model.get_model().mm_projector
        # # VisionTower returns a list of tensors (two last scales)
    # h1 = vt.register_forward_hook(hook_shape("vision_tower"))
    # # Projector returns the fused vision tokens (B, 256, hidden)
    # h2 = proj.register_forward_hook(hook_shape("mm_projector"))
        
    rank0_print("=" * 20 + " Dataset preparation " + "=" * 20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)

    if model_args.tune_mm_mlp_adapter:
        train_dataset = QADatasets(data_args, tokenizer, mode="train")
    else:
        train_dataset = QADatasets(data_args, tokenizer, mode="train")
        
    # sample = train_dataset[0]
    # ids = sample["input_id"]
    # img_tok = tokenizer.convert_tokens_to_ids("<im_patch>")
    # n_img_tokens = (ids == img_tok).sum().item()
    # print(f"#<im_patch> in sample[0]:", n_img_tokens, " expected:", data_args.proj_out_num)
    # print("Decoded prompt (truncated):")
    # print(tokenizer.decode(ids[:300], skip_special_tokens=False))
    
    # sample = train_dataset[1]
    # ids = sample["input_id"].tolist()
    # print(len(ids))
    # print("First token IDs: ", ids)
    # print("Full prompt+answer:")
    # print(tokenizer.decode(
    #     ids,
    #     skip_special_tokens=False,    # so you see the <im_patch> tokens too
    #     clean_up_tokenization_spaces=True
    # ))
    
    # print("── RAW SAMPLE ──")
    # # depending on how VQADataset is written, you may have keys like "question", "answer", etc.
    # for k,v in sample.items():
    #     # skip printing the entire image array, but show its shape
    #     if isinstance(v, torch.Tensor) and v.ndim > 1:
    #         print(f"{k:12s}:", v.shape)
    #     else:
    #         print(v.shape)
    #         print(f"{k:12s}:", v)
        
    # import sys; sys.exit(0)
    
    eval_dataset  = QADatasets(data_args, tokenizer, mode="validation")
    data_collator = DataCollator()
    
    device = torch.device("cuda")
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, 
                               torch.nn.BatchNorm2d,
                               torch.nn.BatchNorm3d)):
            module.running_mean = module.running_mean.to(device)
            module.running_var  = module.running_var.to(device)
        
    # model = model.to(device)
    # sample = train_dataset[0]
    # batch  = data_collator([sample])
    # move to device
    # device = torch.device("cuda")
    # for k in ("images","input_ids","attention_mask","labels"):
    #     batch[k] = batch[k].to(device)
    
    # # call the fusion util
    # (
    #     fake_input_ids,
    #     fake_pos_ids,
    #     fake_attn_mask,
    #     fake_past,
    #     inputs_embeds,
    #     fake_labels,
    # ) = model.prepare_inputs_for_multimodal(
    #     batch["input_ids"],
    #     None,
    #     batch["attention_mask"],
    #     None,
    #     batch["labels"],
    #     batch["images"],
    # )
    
    # print("🤖 multimodal embeds shape:", inputs_embeds.shape)

    # def attach_grad_probes(model):
    #     hooks = []
    #     def mk(name):
    #         def _hook(mod, _, grad_out):
    #             if grad_out and grad_out[0] is not None:
    #                 g = grad_out[0]
    #                 print(f"[grad] {name}: shape={tuple(g.shape)}  norm={g.norm().item():.4f}")
    #         return _hook
    
    #     # projector always trains in your setup
    #     hooks.append(model.get_model().mm_projector.register_full_backward_hook(mk("mm_projector")))
    
    #     # optionally probe a vision block if it's not frozen
    #     vt = model.get_model().get_vision_tower()
    #     if any(p.requires_grad for p in vt.parameters()):
    #         # grab the first leaf module with params
    #         for n, m in vt.named_modules():
    #             if any(p.requires_grad for p in m.parameters(recurse=False)):
    #                 hooks.append(m.register_full_backward_hook(mk(f"vision:{n}")))
    #                 break
    #     return hooks
    
    # # before trainer.train():
    # grad_hooks = attach_grad_probes(model)
        
    rank0_print("=" * 20 + " Training " + "=" * 20)
    trainer = MLLMTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=25)]
    )

    if is_rank_zero():
        wandb.login()
        wandb.init(project="MLLM", name=model_args.wb_name)
        
    if os.path.exists(training_args.output_dir):
        checkpoints = sorted(
            [
                d
                for d in os.listdir(training_args.output_dir)
                if d.startswith("checkpoint-")
                and os.path.isdir(os.path.join(training_args.output_dir, d))
            ],
            key=lambda x: int(x.split("-")[-1]) if "-" in x else 0,
        )
        if checkpoints:
            last_checkpoint = checkpoints[-1]
            resume_ckpt = os.path.join(training_args.output_dir, last_checkpoint)
            rank0_print(f"Resuming from checkpoint: {resume_ckpt}")
            trainer.train(resume_from_checkpoint=resume_ckpt)
        else:
            trainer.train()
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    rank0_print("=" * 20 + " Save model " + "=" * 20)

    def is_lora_param(n: str) -> bool:
        return "lora_" in n
    
    if training_args.lora_enable:

        state_dict_with_lora = model.state_dict()
        torch.save(
            state_dict_with_lora,
            os.path.join(training_args.output_dir, "model_with_lora.bin"),
        )
        
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )

    # --- WRITE RUN SUMMARY FOR GRID SEARCH ---
    if is_rank_zero():
        summary = {
            "best_metric_name": training_args.metric_for_best_model,
            "best_metric_value": float(trainer.state.best_metric) if trainer.state.best_metric is not None else None,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "global_step": trainer.state.global_step,
            "hparams": {
                "learning_rate": training_args.learning_rate,
                "warmup_ratio": training_args.warmup_ratio,
                "weight_decay": training_args.weight_decay,
                "max_grad_norm": training_args.max_grad_norm,
                "label_smoothing_factor": training_args.label_smoothing_factor,
                "lr_scheduler_type": training_args.lr_scheduler_type,
                "lora_enable": training_args.lora_enable,
                "lora_r": training_args.lora_r,
                "lora_alpha": training_args.lora_alpha,
                "lora_dropout": training_args.lora_dropout,
                "model_max_length": training_args.model_max_length,
                "num_train_epochs": training_args.num_train_epochs,
                "seed": training_args.seed,
                "vision_select_layer": model_args.vision_select_layer,
                "mm_projector_type": model_args.mm_projector_type,
            },
        }
        with open(os.path.join(training_args.output_dir, "run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        rank0_print(f"[RUN SUMMARY] best={summary['best_metric_value']} ({summary['best_metric_name']})")

    
    if is_rank_zero():
        wandb.finish()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()