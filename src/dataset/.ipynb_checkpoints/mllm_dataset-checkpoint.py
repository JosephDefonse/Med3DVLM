import os
import json
import torch
import random
import numpy as np
import monai.transforms as mtf
import SimpleITK as sitk
import pandas as pd
from monai.data import set_track_meta
from torch.utils.data import Dataset, ConcatDataset
from src.dataset.prompt_templates import Caption_templates

from monai.transforms import Compose, RandAffine, RandFlip, ToTensor
from monai.transforms import (
    Compose, ScaleIntensityRangePercentiles, NormalizeIntensity, ToTensor,
    Rand3DElastic, RandGaussianSmooth, RandCoarseDropout, RandRotate90, RandZoom
)

class RandomCTWindowJitter:
    """
    Jitters CT window/level after base_norm.
    Assumes input ~[0,1]. We apply a small affine to simulate WL/WC changes.
    """
    def __init__(self, p=0.5, level_delta=0.05, width_scale=(0.95, 1.05)):
        self.p = p
        self.level_delta = level_delta
        self.width_scale = width_scale

    def __call__(self, img):
        # img is a numpy/torch array shaped (1, D, H, W) in ~[0,1]
        import numpy as np
        if np.random.rand() > self.p:
            return img
        lvl_shift = (np.random.rand() * 2 - 1) * self.level_delta   # ±level_delta around mid-gray
        w_scale  = np.random.uniform(*self.width_scale)
        out = (img - 0.5 + lvl_shift) * w_scale + 0.5
        # clamp to [0,1]
        if hasattr(out, "clip"):
            return out.clip(0.0, 1.0)
        else:
            import numpy as np
            return np.clip(out, 0.0, 1.0)

def resample_volume(img: sitk.Image, target_size=(256,256,128)) -> np.ndarray:
    """
    Resample a sitk.Image to the given (W,H,D)=target_size
    and return a numpy array of shape (D,H,W).
    """

    img = sitk.DICOMOrient(img, 'LAS')
    
    orig_size    = img.GetSize()    # (W,H,D)
    # print("orig_size: "+str(orig_size))
    orig_spacing = img.GetSpacing() # (sx,sy,sz)
    # print("orig_spacing: "+str(orig_spacing))
    
    # compute new spacing so physical extent is preserved
    new_size    = list(target_size)
    # print("new_size: "+str(new_size))
    new_spacing = [
        (orig_size[i] * orig_spacing[i]) / new_size[i]
        for i in range(3)
    ]
    # print("new_spacing: "+str(new_spacing))
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)

    origin = img.GetOrigin()
    # print("origin: "+str(origin))
    resampler.SetOutputOrigin(origin)
    
    direction = img.GetDirection()
    # print("direction: "+str(direction))
    resampler.SetOutputDirection(direction)
    
    resampler.SetInterpolator(sitk.sitkLinear)
    # resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    
    out_img = resampler.Execute(img)           # still (W,H,D)
    out_np  = sitk.GetArrayFromImage(out_img)  # numpy shape (D,H,W)
    return out_np

class VQADataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended

        self.image_tokens = "<im_patch>" * args.proj_out_num

        # tok = self.tokenizer.tokenize(self.image_tokens)
        # print(len(tok), tok[:5], tok[-5:])
        
        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_data_train_path)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_data_test_path)
        else:
            print("The mode is not desired ! ")

        # train_transform = Compose([            
        #     RandAffine(
        #         prob=1.0,
        #         translate_range=(0.05, 0.05, 0.05),   # 5% shift
        #         rotate_range=(np.deg2rad(3),)*3,      # ±3°
        #         scale_range=(0.1,)*3,                 # ±10%
        #         mode='bilinear',
        #     ),
        #     RandFlip(prob=0.15, spatial_axis=0),
        #     RandFlip(prob=0.15, spatial_axis=1),
        
        #     RandScaleIntensity(factors=0.1, prob=0.3),
        #     RandShiftIntensity(offsets=0.1, prob=0.3),
        #     RandGaussianNoise(prob=0.1, mean=0.0, std=0.01),
        #     RandAdjustContrast(prob=0.25, gamma=(0.9, 1.1)),
        
        #     ToTensor(dtype=torch.float),
        # ])

        # val_transform = mtf.Compose(
        #     [
        #         ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
        #         NormalizeIntensity(nonzero=True, channel_wise=True),
                
        #         mtf.ToTensor(dtype=torch.float),
        #     ]
        # )

        base_norm = mtf.Compose([
            ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
            NormalizeIntensity(nonzero=True, channel_wise=True),
        ])

        self.base_norm = base_norm

        # Originally, we had in this format
        # SAG = 0
        # COR = 1
        # AX = 2
        # Now because we reformatted everything, it is:
        # AX = 0
        # COR = 1
        # SAG = 2
        
        self.light_train = mtf.Compose([
            self.base_norm,
            mtf.RandAffine(
                prob=0.2,
                translate_range=(0.02, 0.02, 0.02),
                rotate_range=(np.deg2rad(2),)*3,
                scale_range=(0.05,)*3,
                mode='bilinear',
                padding_mode='zeros'
            ),
            mtf.RandShiftIntensity(offsets=0.05, prob=0.3),
            mtf.RandAdjustContrast(prob=0.2, gamma=(0.9, 1.1)),
            mtf.ToTensor(dtype=torch.float),
        ])
        
        self.heavy_train = mtf.Compose([
            self.base_norm,
            mtf.RandAffine(
                prob=0.7,
                translate_range=(0.08, 0.08, 0.08),
                rotate_range=(np.deg2rad(7),)*3,
                scale_range=(0.12,)*3,
                mode='bilinear',
                padding_mode='zeros'
            ),
            mtf.RandFlip(prob=0.5, spatial_axis=2),  # sagittal?
            mtf.RandFlip(prob=0.5, spatial_axis=1),  # coronal?
            mtf.RandShiftIntensity(offsets=0.10, prob=0.5),
            mtf.RandAdjustContrast(prob=0.4, gamma=(0.8, 1.25)),
            mtf.RandGaussianNoise(prob=0.25, mean=0.0, std=0.01),
            mtf.ToTensor(dtype=torch.float),
        ])
        
        val_transform = Compose([
            base_norm,
            mtf.ToTensor(dtype=torch.float),
        ])
        
        set_track_meta(False)

        if mode == "train":
            self.transform = None
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]

                is_dup = int(data.get("IsDup", 0))

                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                # image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized
                # image = sitk.ReadImage(image_abs_path)
                # image = sitk.GetArrayFromImage(image)
                # image = np.expand_dims(image, axis=0)
                # image = self.transform(image)

                img_sitk = sitk.ReadImage(image_abs_path)
                
                # Print original image info
                # orig_size = img_sitk.GetSize()         # (W, H, D)
                # orig_spacing = img_sitk.GetSpacing()   # (sx, sy, sz)
                # print(f"Original size: {orig_size}, spacing: {orig_spacing}")
                                
                vol_np   = resample_volume(img_sitk, target_size=(256,256,128))

                # Print resampled shape
                # print(f"Resampled shape (D,H,W): {vol_np.shape}")

                # MONAI transforms requires the following
                # “Most of the image transformations in monai.transforms assume the input image is in the channel-first format, which has the shape (num_channels, spatial_dim_1[, spatial_dim_2, ...]).”
                # Given that we are doing below, it creates: (1, D, H, W)/(1, H, D, W)/(1, H, W, D)
                # This is correct because CT scan is 1 channel (grey scale)
                image    = np.expand_dims(vol_np, axis=0)

                # print(f"After adding channel dim: {image.shape}")
                
                # image    = self.transform(image)

                if self.mode == "train":
                    tr = self.heavy_train if is_dup == 1 else self.light_train
                    image = tr(image)
                else:
                    image = self.transform(image)
                
                # print(f"After adding channel dim: {image.shape}")

                if self.close_ended:
                    question = data["Question"]
                    choices = "Choices: A. {} B. {} C. {} D. {}".format(
                        data["Choice A"],
                        data["Choice B"],
                        data["Choice C"],
                        data["Choice D"],
                        
                    )
                    question = question + " " + choices
                    # answer = "{}. {}".format(data["Answer Choice"], data["Answer"])
                    answer   = data["Answer Choice"]
                else:
                    question = data["Question"]
                    answer = str(data["Answer"])

                question = data['Text'] + " " + self.image_tokens + " " + question
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                # if self.mode in ("validation", "val", "test"):
                #     image = torch.zeros_like(image)
                
                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "answer_choice": data["Answer Choice"],
                    "question_type": data["Question Type"],
                }
                
                # if self.args.seg_enable:
                #     ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class VQAYNDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_yn_data_train_path)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_yn_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_yn_data_test_path)
        else:
            print("The mode is not desired ! ")

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                # image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized
                image = sitk.ReadImage(image_abs_path)
                image = sitk.GetArrayFromImage(image)
                image = np.expand_dims(image, axis=0)
                image = self.transform(image)

                question = data["Question"]
                answer = str(data["Answer"])

                question = self.image_tokens + " " + question
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "answer_choice": data["Answer Choice"],
                    "question_type": data["Question Type"],
                }
                # if self.args.seg_enable:
                #     ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class TextDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(TextDatasets, self).__init__()
        self.ds_list = [
            # CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            VQADataset(args, tokenizer, close_ended=False, mode=mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class QADatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(QADatasets, self).__init__()
        self.ds_list = [
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            # VQADataset(args, tokenizer, close_ended=False, mode=mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class TextYNDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(TextYNDatasets, self).__init__()
        self.ds_list = [
            # CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            VQADataset(args, tokenizer, close_ended=False, mode=mode),
            VQAYNDataset(args, tokenizer, mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]