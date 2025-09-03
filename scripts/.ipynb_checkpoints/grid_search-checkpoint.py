#!/usr/bin/env python3
import os, re, json, itertools, csv, sys
from pathlib import Path
import subprocess

HF_HOME = "/home/sdef0001/iq38_scratch/nmdid/cache/huggingface"
TRITON_CACHE_DIR = "/home/sdef0001/iq38_scratch/nmdid/cache/triton"

BASE_ENV = os.environ.copy()
BASE_ENV["PYTHONPATH"] = str(Path.cwd())
BASE_ENV.setdefault("HF_HOME", HF_HOME)
BASE_ENV.setdefault("TRANSFORMERS_CACHE", HF_HOME)
BASE_ENV.setdefault("TRITON_CACHE_DIR", TRITON_CACHE_DIR)
BASE_ENV.setdefault("PYTHONUNBUFFERED", "1")

OUT_ROOT = Path("./output/GS_runs_v2")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_ROOT / "grid_results.csv"
BEST_LINK = Path("./output/GS_BEST")  # symlink to best run

# ---- grid (24 runs) ----
lrs                 = [3e-5]
warmups             = [0.06, 0.10]
weight_decays       = [0.01, 0.5, 0.1]
label_smoothings    = [0.05, 0.1, 0.2]

# fixed (best-so-far neighborhood)
lora_r = 64
lora_alpha = 128
lora_dropout = 0.10
max_grad_norms = [0.5, 1.0]  # try [0.5, 1.0] later if still stuck

def tag_for(hp):
    lr = f"{hp['lr']:.0e}".replace("+","")
    wr = int(round(hp["warmup_ratio"]*100))
    wd = f"{hp['weight_decay']:.3f}".replace("0.", "")
    ls = f"{hp['label_smoothing']:.2f}".replace("0.", "")
    mgn = f"{hp['max_grad_norm']:.1f}".replace(".","p")
    return f"r{lora_r}_a{lora_alpha}_p{int(lora_dropout*100):03d}_lr{lr}_wr{wr:03d}_wd{wd}_ls{ls}_mgn{mgn}"

def harvest_trainer_state(out_dir: Path):
    p = out_dir / "trainer_state.json"
    if not p.exists(): return None, None
    try:
        st = json.loads(p.read_text())
        return st.get("best_metric", None), st.get("best_model_checkpoint", None)
    except Exception:
        return None, None

def harvest_from_log(log_path: Path):
    if not log_path.exists(): return None
    pat = re.compile(r"eval_choice_acc[\"']?\s*:\s*([0-9]*\.?[0-9]+)")
    best = None
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                v = float(m.group(1))
                best = v if (best is None or v > best) else best
    return best

def append_csv_row(row: dict):
    header = [
        "run_tag","output_dir","best_metric","best_model_checkpoint",
        "learning_rate","warmup_ratio","weight_decay","label_smoothing",
        "max_grad_norm","lora_r","lora_alpha","lora_dropout"
    ]
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header: w.writeheader()
        w.writerow({k: row.get(k) for k in header})

def update_best_symlink(best_out_dir: Path):
    try:
        if BEST_LINK.exists() or BEST_LINK.is_symlink():
            BEST_LINK.unlink()
        BEST_LINK.symlink_to(best_out_dir.resolve())
    except Exception as e:
        print(f"[WARN] Could not update best symlink: {e}")

def run_once(hp):
    tag = tag_for(hp)
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    cmd = [
        "deepspeed", "src/train/train_vlm.py",
        "--deepspeed", "./scripts/zero9.json",
        "--vision_tower", "dcformer",
        "--model_name_or_path", "Qwen/Qwen2.5-7B-Instruct",
        "--model_type", "vlm_qwen",
        "--mm_projector_type", "mixer",
        "--lora_enable", "True",
        "--vision_select_layer", "-2",
        "--pretrain_vision_model", "./output/DCFormer_SigLIP/pretrained_ViT.bin",
        "--pretrain_mm_mlp_adapter", "./output/Med3DVLM-Qwen-2.5-7B/mm_projector.bin",
        "--data_root", "/home/sdef0001/iq38_scratch/nmdid",
        "--vqa_data_train_path", "/home/sdef0001/iq38_scratch/nmdid/nmdid_cause_vqa_8_train.csv",
        "--vqa_data_val_path",   "/home/sdef0001/iq38_scratch/nmdid/nmdid_cause_vqa_8_val.csv",
        "--vqa_data_test_path",  "/home/sdef0001/iq38_scratch/nmdid/nmdid_cause_vqa_8_test.csv",
        "--bf16", "True",
        "--per_device_train_batch_size", "8",
        "--per_device_eval_batch_size", "4",
        "--gradient_accumulation_steps", "1",
        "--eval_strategy", "steps",
        "--eval_steps", "50",
        "--metric_for_best_model", "eval_choice_acc",
        "--greater_is_better", "True",
        "--load_best_model_at_end", "True",
        "--save_strategy", "steps",
        "--save_steps", "50",
        "--save_total_limit", "3",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "10",             # <= integer to fix W&B warnings
        "--gradient_checkpointing", "True",
        "--dataloader_pin_memory", "True",
        "--dataloader_num_workers", "4",
        "--report_to", "wandb",
        "--num_train_epochs", "5",
        "--seed", "42",

        "--wb_name", f"GSv2_{tag}",
        "--output_dir", str(out_dir),

        "--learning_rate", str(hp["lr"]),
        "--warmup_ratio", str(hp["warmup_ratio"]),
        "--weight_decay", str(hp["weight_decay"]),
        "--label_smoothing_factor", str(hp["label_smoothing"]),
        "--max_grad_norm", str(hp["max_grad_norm"]),

        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", str(lora_dropout),
    ]

    print(f"\n=== RUN {tag} ===")
    print(" ".join(cmd))

    with open(log_path, "w", buffering=1) as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env=BASE_ENV)
        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
        rc = p.wait()

    if rc != 0:
        print(f"[{tag}] exited with code {rc}")

    best_metric, best_ckpt = harvest_trainer_state(out_dir)
    if best_metric is None:
        best_metric = harvest_from_log(log_path)
    if best_metric is None:
        return None

    return {
        "run_tag": tag,
        "output_dir": str(out_dir),
        "best_metric": float(best_metric),
        "best_model_checkpoint": str(best_ckpt) if best_ckpt else "",
        "learning_rate": hp["lr"],
        "warmup_ratio": hp["warmup_ratio"],
        "weight_decay": hp["weight_decay"],
        "label_smoothing": hp["label_smoothing"],
        "max_grad_norm": hp["max_grad_norm"],
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
    }

def main():
    combos = []
    for lr, wr, wd, ls, mgn in itertools.product(
        lrs, warmups, weight_decays, label_smoothings, max_grad_norms
    ):
        combos.append({
            "lr": lr,
            "warmup_ratio": wr,
            "weight_decay": wd,
            "label_smoothing": ls,
            "max_grad_norm": mgn,
        })
    print(f"Total runs: {len(combos)}")

    best = None
    for hp in combos:
        res = run_once(hp)
        if not res: continue
        append_csv_row(res)
        if (best is None) or (res["best_metric"] > best["best_metric"]):
            best = res
            update_best_symlink(Path(best["output_dir"]))
            print(f"💡 NEW BEST: acc={best['best_metric']:.4f}  tag={best['run_tag']}  dir={best['output_dir']}")

    if best:
        print("\n=== BEST OVERALL ===")
        print(f"eval_choice_acc = {best['best_metric']:.4f}")
        print(f"run_tag         = {best['run_tag']}")
        print(f"output_dir      = {best['output_dir']}")
        print(f"(symlinked at {BEST_LINK})")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    main()
