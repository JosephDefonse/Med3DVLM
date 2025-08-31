#!/usr/bin/env python3
import os, re, json, itertools, csv, sys
from pathlib import Path
import subprocess

# ---------- ENV like your .sh ----------
HF_HOME = "/home/sdef0001/iq38_scratch/nmdid/cache/huggingface"
TRITON_CACHE_DIR = "/home/sdef0001/iq38_scratch/nmdid/cache/triton"

BASE_ENV = os.environ.copy()
BASE_ENV["PYTHONPATH"] = str(Path.cwd())
BASE_ENV.setdefault("HF_HOME", HF_HOME)
BASE_ENV.setdefault("TRANSFORMERS_CACHE", HF_HOME)  # transformers warns but still respects HF_HOME
BASE_ENV.setdefault("TRITON_CACHE_DIR", TRITON_CACHE_DIR)
BASE_ENV.setdefault("PYTHONUNBUFFERED", "1")
# Uncomment if you want guaranteed W&B online naming (or set WANDB_MODE=offline)
# BASE_ENV["WANDB_PROJECT"] = "MLLM"

OUT_ROOT = Path("./output/GS_runs")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_ROOT / "grid_results.csv"
BEST_LINK = Path("./output/GS_BEST")  # symlink -> best run’s output_dir

# ---------- GRID ----------
lrs           = [2e-5, 3e-5, 5e-5]
warmups       = [0.06, 0.10]
lora_rs       = [32, 64]
lora_dropouts = [0.05, 0.10]
# keep alpha modest: alpha = 2 * r
def alpha_for(r): return 2 * r

def fmt_lr(x: float) -> str:
    # readable tag for lr
    s = f"{x:.0e}"
    # keep '2e-05' style (no plus sign)
    return s.replace("+", "")

def tag_for(hp: dict) -> str:
    # p005 for 0.05; wr006 for 0.06
    p   = int(round(hp["lora_dropout"] * 100))
    wr  = int(round(hp["warmup_ratio"] * 100))
    lr  = fmt_lr(hp["lr"])
    return f"r{hp['lora_r']}_a{hp['lora_alpha']}_p{p:03d}_lr{lr}_wr{wr:03d}"

def harvest_from_trainer_state(out_dir: Path):
    state_p = out_dir / "trainer_state.json"
    if not state_p.exists():
        return None, None
    try:
        with open(state_p, "r") as f:
            st = json.load(f)
        return st.get("best_metric", None), st.get("best_model_checkpoint", None)
    except Exception:
        return None, None

def harvest_from_log(log_path: Path):
    if not log_path.exists():
        return None
    # Parse max eval_choice_acc from Trainer prints like: {'eval_choice_acc': 0.60317, ...}
    pat = re.compile(r"eval_choice_acc[\"']?\s*:\s*([0-9]*\.?[0-9]+)")
    best = None
    try:
        with open(log_path, "r", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    val = float(m.group(1))
                    best = val if (best is None or val > best) else best
    except Exception:
        pass
    return best

def append_csv_row(row: dict):
    header = [
        "run_tag","output_dir","best_metric","best_model_checkpoint",
        "learning_rate","warmup_ratio","lora_r","lora_alpha","lora_dropout"
    ]
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        # ensure only header keys get written
        w.writerow({k: row.get(k) for k in header})

def update_best_symlink(best_out_dir: Path):
    try:
        if BEST_LINK.exists() or BEST_LINK.is_symlink():
            BEST_LINK.unlink()
        BEST_LINK.symlink_to(best_out_dir.resolve())
    except Exception as e:
        print(f"[WARN] Could not update best symlink: {e}")

def run_once(hp: dict) -> dict | None:
    tag = tag_for(hp)
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    cmd = [
        "deepspeed",
        "src/train/train_vlm.py",  # user script
        "--deepspeed", "./scripts/zero9.json",

        # fixed args from your .sh
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
        "--logging_steps", "0.001",
        "--gradient_checkpointing", "True",
        "--dataloader_pin_memory", "True",
        "--dataloader_num_workers", "4",
        "--report_to", "wandb",
        "--num_train_epochs", "5",
        "--weight_decay", "0.0",
        "--label_smoothing_factor", "0.0",
        "--max_grad_norm", "1.0",
        "--seed", "42",

        # per-run vars
        "--wb_name", f"GS_{tag}",
        "--output_dir", str(out_dir),
        "--learning_rate", str(hp["lr"]),
        "--warmup_ratio", str(hp["warmup_ratio"]),
        "--lora_r", str(hp["lora_r"]),
        "--lora_alpha", str(hp["lora_alpha"]),
        "--lora_dropout", str(hp["lora_dropout"]),
    ]

    print(f"\n=== RUN {tag} ===")
    print(" ".join(cmd))

    # Stream logs live to console and file
    with open(log_path, "w", buffering=1) as lf:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=BASE_ENV
        )
        for line in p.stdout:
            sys.stdout.write(line)
            lf.write(line)
        rc = p.wait()

    if rc != 0:
        print(f"[{tag}] exited with code {rc}")
        # Try to harvest what we can anyway
    best_metric, best_ckpt = harvest_from_trainer_state(out_dir)
    if best_metric is None:
        best_metric = harvest_from_log(log_path)

    if best_metric is None:
        # failed run; return minimal info so main() can skip
        return None

    return {
        "run_tag": tag,
        "output_dir": str(out_dir),
        "best_metric": float(best_metric),
        "best_model_checkpoint": str(best_ckpt) if best_ckpt else "",
        "learning_rate": hp["lr"],
        "warmup_ratio": hp["warmup_ratio"],
        "lora_r": hp["lora_r"],
        "lora_alpha": hp["lora_alpha"],
        "lora_dropout": hp["lora_dropout"],
    }

def main():
    combos = []
    for r, p, lr, wr in itertools.product(lora_rs, lora_dropouts, lrs, warmups):
        combos.append({
            "lora_r": r,
            "lora_alpha": alpha_for(r),
            "lora_dropout": p,
            "lr": lr,
            "warmup_ratio": wr,
        })

    print(f"Total runs: {len(combos)}")
    best = None

    for hp in combos:
        res = run_once(hp)
        if not res:
            continue
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