import os
import random
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

def resample_volume(img: sitk.Image, target_size=(256,256,128)) -> np.ndarray:
    """
    Resample a sitk.Image to the given (W,H,D)=target_size
    and return a numpy array of shape (D,H,W).
    """
    orig_size    = img.GetSize()    # (W,H,D)
    orig_spacing = img.GetSpacing() # (sx,sy,sz)
    
    # compute new spacing so physical extent is preserved
    new_size    = list(target_size)
    new_spacing = [
        (orig_size[i] * orig_spacing[i]) / new_size[i]
        for i in range(3)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    
    out_img = resampler.Execute(img)           # still (W,H,D)
    out_np  = sitk.GetArrayFromImage(out_img)  # numpy shape (D,H,W)
    return out_np

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


@dataclass
class AllArguments:
    model_name_or_path: str = field(default="./Med3DVLM-Qwen-2.5-7B")

    image_path: str = field(
        default="./data/demo/024421/Axial_C__portal_venous_phase.nii.gz"
    )

    question: str = field(
        default="Describe the findings of the medical image you see.",
        metadata={"help": "Question to ask the model."},
    )

    model_max_length: int = field(
        default=512, metadata={"help": "Maximum length of the input sequence."}
    )


def main():
    seed_everything(42)
    device = torch.device("cuda")  # 'cpu', 'cuda'
    dtype = torch.bfloat16  # or bfloat16, float16, float32

    parser = transformers.HfArgumentParser(AllArguments)
    args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    proj_out_num = (
        model.get_model().config.proj_out_num
        if hasattr(model.get_model().config, "proj_out_num")
        else 256
    )

    question = args.question

    image_tokens = "<im_patch>" * proj_out_num
    input_txt = image_tokens + question
    input_id = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device=device)

    # image_np = np.expand_dims(
    #     sitk.GetArrayFromImage(sitk.ReadImage(args.image_path)), axis=0
    # )

    # --- insert preprocessing here ---
    # read as SimpleITK image
    sitk_img = sitk.ReadImage(args.image_path)
    # resample to W=256,H=256,D=128
    vol_np   = resample_volume(sitk_img, target_size=(256,256,128))
    # add channel + batch dims: (1,1,128,256,256)
    image_np = vol_np[np.newaxis, ...]
    # ---
    
    image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)

    generation = model.generate(
        images=image_pt,
        inputs=input_id,
        max_new_tokens=args.model_max_length,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
    )

    generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

    print("question: ", question)
    print("generated_texts: ", generated_texts[0])


if __name__ == "__main__":
    main()
