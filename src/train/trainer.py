import os

import torch
from torch import distributed as dist
from tqdm import tqdm
from transformers import Trainer

from torch.utils.data import DataLoader, WeightedRandomSampler

import wandb


def is_rank_zero():
    if "RANK" in os.environ:
        if int(os.environ["RANK"]) != 0:
            return False
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() != 0:
            return False
    return True


class CLIPTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)
        loss = outputs["loss"]

        if is_rank_zero():
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "train/step": self.state.global_step,
                },
                step=self.state.global_step,
            )

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["loss"]
            logits = outputs["logits"]

            if prediction_loss_only:
                return (loss, None, None)

            labels = inputs["labels"]
            return (loss, logits, labels)


class MLLMTrainer(Trainer):

    # def get_train_dataloader(self) -> DataLoader:
    #     """
    #     Overrides the default training dataloaloader to use a WeightedRandomSampler,
    #     which addresses the severe class imbalance in the dataset.
    #     """
    #     if self.train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")

    #     # 1. Calculate class weights from the actual training dataframe
    #     train_df = self.train_dataset.ds_list[0].data_list
    #     class_counts = train_df['Answer'].value_counts().to_dict()
    #     num_samples = len(train_df)
        
    #     # Calculate weight for each class: weight = total_samples / class_count
    #     class_weights = {
    #         cls: num_samples / count for cls, count in class_counts.items()
    #     }

    #     # 2. Assign a weight to every single sample in the dataset
    #     train_labels = train_df['Answer']
    #     sample_weights = torch.tensor([class_weights[label] for label in train_labels])

    #     # 3. Create the WeightedRandomSampler
    #     sampler = WeightedRandomSampler(
    #         weights=sample_weights,
    #         num_samples=len(sample_weights),
    #         replacement=True
    #     )

    #     # 4. Manually create the DataLoader instead of calling super()
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self._train_batch_size,
    #         sampler=sampler,
    #         collate_fn=self.data_collator,
    #         drop_last=self.args.dataloader_drop_last,
    #         num_workers=self.args.dataloader_num_workers,
    #         pin_memory=self.args.dataloader_pin_memory,
    #     )
    
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        outputs = model(**inputs)
        loss = outputs["loss"]

        if is_rank_zero():
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "train/step": self.state.global_step,
                },
                step=self.state.global_step,
            )

        return (loss, outputs) if return_outputs else loss
