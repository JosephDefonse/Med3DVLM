#!/usr/bin/env python3
import argparse
import os

import SimpleITK as sitk
import numpy as np
import torch
import monai.transforms as mtf
from transformers import AutoTokenizer

from src.model.llm import VLMQwenForCausalLM

# 1) Resample exactly as in your VQADataset
def resample_volume(img: sitk.Image, target_size=(256,256,128)) -> np.ndarray:
    orig_size    = img.GetSize()    # (W,H,D)
    orig_spacing = img.GetSpacing() # (sx,sy,sz)
    new_size     = list(target_size)
    new_spacing  = [(orig_size[i] * orig_spacing[i]) / new_size[i] for i in range(3)]
    resampler    = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    out_img = resampler.Execute(img)            # still (W,H,D)
    out_np  = sitk.GetArrayFromImage(out_img)   # numpy (D,H,W)
    return out_np

# 2) Build the same “val” transform you used in VQADataset
val_transform = mtf.Compose([
    mtf.ToTensor(dtype=torch.float),   # -> Tensor shape (1, D, H, W)
])

def predict_single(image_path, model_dir, proj_out_num=256, max_length=512, device="cuda"):
    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    model = VLMQwenForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map="auto",
    )
    model.to(device)

    # 3) Read + preprocess the volume
    sitk_img = sitk.ReadImage(image_path)
    vol_np   = resample_volume(sitk_img, target_size=(256,256,128))
    # expand to (C=1, D, H, W) and transform
    image_tensor = val_transform(vol_np[np.newaxis, ...]).to(device)

    # 4) Build your single-case prompt
    image_tokens  = "<im_patch>" * proj_out_num
    QUESTION_TEXT = (
        "What is the cause of death? " 
        "Choices: A. Asphyxia  B. Blood disorders  C. Cardiac arrhythmia  D. Cerebrovascular  E. Chronic obstructive pulmonary disease  F. Drowning  G. Emboli  H. Ethanol intoxication  I. Ethanolism  J. Exposure  K. Gaserointestinal hemorrhage  L. Gunshot wound  M. Hanging  N. Head and neck injuries  O. Hepatic failure  P. Hypertension  Q. Malnutrition  R. Multiple injuries  S. Natural  T. Obesity  U. Pneumonia  V. Renal failure  W. Respiratory Distress Syndrome  X. Sepsis  Y. Substance intoxication  Z. TBD\n"
        "(please respond with ONLY the single letter)\n\n"
        "Please respond exactly like this:\n"
        "The answer is:\n"
    )
    prompt = image_tokens + " " + QUESTION_TEXT

    # tokenize prompt (no label, since we’re generating)
    tokenized = tokenizer(
        prompt,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    # 5) Generate
    with torch.inference_mode():
        generated = model.generate(
            image_tensor.unsqueeze(0),               # add batch dim
            tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            temperature=1.0,
            max_new_tokens=10,
            do_sample=False,
            top_p=None,
            top_k=0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("."),
        )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    return decoded

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-case VQA prediction"
    )
    parser.add_argument("--image_path", required=True,
                        help="Path to your .nii/.dcm file")
    parser.add_argument("--model_dir", required=True,
                        help="Directory of Med3DVLM-Qwen-… model")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda","cpu"])
    args = parser.parse_args()

    pred = predict_single(
        args.image_path,
        args.model_dir,
        device=args.device
    )
    print(f"The answer is: {pred}")