# **Fine-Tuning Biomedical Foundation Models for Post-Mortem Data Analysis in Forensic Medicine**

A research fork of **Med3DVLM** adapted to **post‑mortem computed tomography (PMCT)**. This repository contains code and instructions to reproduce our paper:

> **Fine‑Tuning Biomedical Foundation Models for Post‑Mortem Data Analysis in Forensic Medicine**
> *Shean De Fonseka¹, Christopher Bain¹, Deval Mehta¹, and Richard Bassed¹²*
> ¹ Monash University · ² Victorian Institute of Forensic Medicine

Our goal is to determine whether a modern 3D vision‑language model (Med3DVLM) can be **fine‑tuned** to classify **cause of death (COD)** from PMCT volumes via a **close‑ended VQA** formulation.

---

## Summary

* Task: **COD classification** on PMCT as **four‑option VQA (A-D)**.
* Dataset: **NMDID** (licensed access; not redistributed here).
* Preprocessing: **DICOM to NIfTI**, LAS re‑orientation, **256×256×128** resampling, intensity standardisation.
* Training: **LoRA on LLM** (rank=32, α=64, dropout=0.1), **fully‑trainable vision tower (DCFormer)** and **mm_projector**; conservative 3D augmentations; **duplication‑based class balancing**.
* Results (test set **320 scans**): **62.8% accuracy**, **macro‑F1 50.8%**; best class **Gunshot Wound** (F1 **69.1%**).

---

## Contents

* [Environment](#environment)
* [Installation](#installation)
* [Data Access & Ethics](#data-access--ethics)
* [Preparation (NMDID → NIfTI)](#preparation-nmdid--nifti)
* [Dataset Splits & Balancing](#dataset-splits--balancing)
* [Model & Checkpoints](#model--checkpoints)
* [Training](#training)
* [Evaluation](#evaluation)
* [Inference Demo](#inference-demo)
* [Results Summary](#results-summary)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

---

## Environment

We used the following core versions (others may work):

* Python 3.12
* PyTorch 2.6, torchvision 0.21
* MONAI 1.4
* DeepSpeed 0.16

> Tip: a reproducible `conda` env is provided; or install from `requirements.txt`.

## Installation

```bash
# clone
git clone https://github.com/JosephDefonse/Med3DVLM.git
cd Med3DVLM

# either conda
conda env create -f env.yaml
conda activate Med3DVLM

# or pip
pip install -r requirements.txt

# add project root to pythonpath
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Data Access & Ethics

* **Source**: New Mexico Decedent Image Database (**NMDID**). Access requires a license; **raw data is not redistributed** by this repo.
* **Use**: educational/scientific; no re‑identification or facial reconstruction; follow institutional approvals (Monash/VIFM) and storage on secure HPC.
* **Storage**: all processing performed on Monash **M3 (MASSIVE)** cluster. See paper for governance, retention and compliance details.

## Preparation (NMDID → NIfTI)

1. **Download** licensed NMDID cases locally on your secure environment.
2. **Convert** DICOM stacks to NIfTI (`.nii.gz`) and re‑orient to **LAS**; **resample** to **(256, 256, 128)** with SimpleITK; **standardise** intensities.

### Inclusion/Exclusion (used in our study)

* Adults **(≥18 years)** only; exclude pregnancy.
* Require at least one of: **Head/Brain**, **Chest/Lung**, **Abdomen** series.
* Exclude corrupted series and distal extremities for this task.

### Augmentations (conservative)

* Random affine (±3° rot, ±5% trans, 0.9–1.1 scale), axis flips (p=0.15 each), gamma 0.9–1.1, intensity shift/scale, light Gaussian noise.

## Dataset Splits & Balancing

* **Case‑level** split to avoid leakage across regions from the same decedent: ~**70/15/15** train/val/test (151 cases ≈ 105/23/23).
* The pipeline yields **thousands of scans** from these cases; **test set used for reporting: 320 scans**.
* **Class imbalance** handled with **duplication‑based balancing** on the **training** set: duplicate minority‑class scans up to the majority count; apply augmentations to reduced overfitting. Validation/test remain imbalanced to reflect reality.

## Model & Checkpoints

We fine‑tune **Med3DVLM (Qwen‑2.5‑7B)** for close‑ended VQA:

* **LLM (Qwen‑2.5‑7B)**: base weights **frozen**; **LoRA** on linear layers only (rank **32**, **α=64**, dropout **0.1**).
* **Vision tower (DCFormer)**: **fully trainable** end‑to‑end (initialized from SigLIP‑pretrained checkpoint).
* **Multimodal projector (Mixer)**: **fully trainable**.

### Our PMCT Checkpoint

* **Hugging Face**: [https://huggingface.co/JosephDefonse/Med3DVLM-PMCT](https://huggingface.co/JosephDefonse/Med3DVLM-PMCT)

## Results Summary

* **Test scans**: **320**
* **Overall accuracy**: **62.8%**
* **Macro‑F1**: **50.8%**
* **Best class**: **Gunshot Wound** (**F1 69.1%**)

See the paper for full confusion matrix and per‑class metrics.

## Citation

If you use this code or PMCT checkpoint, please cite our paper and the original Med3DVLM work.

```bibtex
@misc{defonseka2025pmct_med3dvlm,
  title={Fine-Tuning Biomedical Foundation Models for Post-Mortem Data Analysis in Forensic Medicine},
  author={Shean De Fonseka and Christopher Bain and Deval Mehta and Richard Bassed},
  year={2025},
  note={Monash University & Victorian Institute of Forensic Medicine},
  url={https://github.com/JosephDefonse/Med3DVLM}
}

@article{xin2025med3dvlm,
  title={Med3DVLM: An Efficient Vision-Language Model for 3D Medical Image Analysis},
  author={Xin, Yu and Ates, Gorkem Can and Gong, Kuang and Shao, Wei},
  journal={arXiv preprint arXiv:2503.20047},
  year={2025}
}
```

## Acknowledgements

This research builds on **Med3DVLM** and related components (DCFormer, SigLIP, Mixer, Qwen2.5). We thank the **NMDID** administrators, Monash **MASSIVE/M3** HPC support, and colleagues at **VIFM** for guidance.

Original Med3DVLM code: [https://github.com/mirthAI/Med3DVLM](https://github.com/mirthAI/Med3DVLM).
