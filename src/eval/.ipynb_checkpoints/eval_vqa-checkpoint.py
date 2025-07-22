import argparse
import csv
import os
import random

"""
RUN export PYTHONPATH="$PWD" in BASH BEFORE RUNNING
bash scripts/eval/eval_vqa.sh
"""

# If the model is not from huggingface but local, please uncomment and import the model architecture.
# from LaMed.src.model.language_model import *
import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model.llm import VLMQwenForCausalLM
from src.dataset.mllm_dataset import VQADataset
import torch.nn.functional as F

bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="./models/Med3DVLM-Qwen-2.5-7B"
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--vqa_data_test_path", type=str, default="./data/M3D-VQA/M3D_VQA_test.csv"
    )
    parser.add_argument("--close_ended", action="store_true", default=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/eval_vqa/",
    )

    parser.add_argument("--proj_out_num", type=int, default=256)

    return parser.parse_args(args)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    model = VLMQwenForCausalLM.from_pretrained(
        args.model_name_or_path,  # loads vlm_qwen
        trust_remote_code=False   # you already have the class locally
    )
    model = model.to(device=device)

    test_dataset = VQADataset(
        args, tokenizer=tokenizer, close_ended=args.close_ended, mode="test"
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    answer_tokens = [ tokenizer.convert_tokens_to_ids(l) for l in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") ]

    for letter in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        tid = tokenizer.convert_tokens_to_ids(letter)
        tok = tokenizer.convert_ids_to_tokens(tid)
        print(f"{letter!r} → id {tid} → token {tok!r}")

    # import sys; sys.exit(0)
    
    from transformers import LogitsProcessorList, MinLengthLogitsProcessor, \
                                 LogitsProcessor, LogitsWarper
    
    class RestrictToAnswers(LogitsProcessor):
        def __call__(self, input_ids, scores):
            # zero out everything except your answer_tokens
            mask = torch.full_like(scores, float("-inf"))
            mask[:, answer_tokens] = scores[:, answer_tokens]
            return mask
    
    logits_processor = LogitsProcessorList([ RestrictToAnswers() ])
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_name = args.model_name_or_path.split("/")[-1]

    if args.close_ended:
        print("Evaluating close-ended VQA...")
        output_path = os.path.join(args.output_dir, f"{model_name}_eval_close_vqa.csv")
        with open(output_path, mode="w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(
                [
                    "Question Type",
                    "Question",
                    "Answer",
                    "Answer Choice",
                    "Pred",
                    "Correct",
                ]
            )
            for sample in tqdm(test_dataloader):
                question = sample["question"]
                question_type = sample["question_type"].item()
                answer_choice = sample["answer_choice"]
                answer = sample["answer"]

                image = sample["image"].to(device=device)

                input_ids      = sample["input_id"].to(device)        # full prompt: Q + Choices + <ANS>
                attention_mask = sample["attention_mask"].to(device)
                images         = sample["image"].to(device)

                # with torch.inference_mode():
                #     generation = model.generate(
                #         image,
                #         input_id,
                #         max_new_tokens=args.max_new_tokens,
                #         do_sample=args.do_sample,
                #         top_p=args.top_p,
                #         temperature=args.temperature,
                #         logits_processor=logits_processor,
                #     )
                # generated_texts = tokenizer.batch_decode(
                #     generation, skip_special_tokens=True
                # )

                # --- instead of generate(), do straight logit argmax over A–Z ---
                with torch.no_grad():
                    outputs = model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )      # forward
                    lm_logits    = outputs.logits[:, -1, :]                    # [1, vocab_size]
                    answer_logits= lm_logits[:, answer_tokens]                 # [1,26]
                
                    # 1) get the raw probs
                    probs        = F.softmax(answer_logits, dim=-1).squeeze(0)  # [26]
                
                    # 2) map back to letters
                    letters      = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                    for letter, p in zip(letters, probs):
                        print(f"{letter}: {p.item():.4f}")
                
                    # 3) pick the top‐1 as before
                    idx    = probs.argmax().item()     # 0–25
                    pred   = letters[idx]
                generated_texts = [pred]
                
                if answer_choice[0] == generated_texts[0]:
                    correct = 1
                else:
                    correct = 0

                writer.writerow(
                    [
                        question_type,
                        question[0],
                        answer[0],
                        answer_choice[0],
                        generated_texts[0],
                        correct,
                    ]
                )
    else:
        print("Evaluating open-ended VQA...")
        output_path = os.path.join(args.output_dir, f"{model_name}_eval_open_vqa.csv")
        with open(output_path, mode="w") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(
                [
                    "Question Type",
                    "Question",
                    "Answer",
                    "Pred",
                    "bleu",
                    "rouge1",
                    "meteor",
                    "bert_f1",
                ]
            )
            for sample in tqdm(test_dataloader):
                question = sample["question"]
                question_type = sample["question_type"].item()
                answer = sample["answer"]

                image = sample["image"].to(device=device)
                input_id = tokenizer(question, return_tensors="pt")["input_ids"].to(
                    device=device
                )

                with torch.inference_mode():
                    generation = model.generate(
                        image,
                        input_id,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                        top_p=args.top_p,
                        temperature=args.temperature,
                    )
                generated_texts = tokenizer.batch_decode(
                    generation, skip_special_tokens=True
                )

                result = dict()
                decoded_preds, decoded_labels = postprocess_text(
                    generated_texts, answer
                )
                bleu_score = bleu.compute(
                    predictions=decoded_preds, references=decoded_labels, max_order=1
                )
                result["bleu"] = bleu_score["bleu"]

                rouge_score = rouge.compute(
                    predictions=decoded_preds,
                    references=decoded_labels,
                    rouge_types=["rouge1"],
                )
                result["rouge1"] = rouge_score["rouge1"]

                meteor_score = meteor.compute(
                    predictions=decoded_preds, references=decoded_labels
                )
                result["meteor"] = meteor_score["meteor"]

                bert_score = bertscore.compute(
                    predictions=decoded_preds, references=decoded_labels, lang="en"
                )
                result["bert_f1"] = sum(bert_score["f1"]) / len(bert_score["f1"])

                writer.writerow(
                    [
                        question_type,
                        question[0],
                        answer[0],
                        generated_texts[0],
                        result["bleu"],
                        result["rouge1"],
                        result["meteor"],
                        result["bert_f1"],
                    ]
                )

    Qustion_Type = {1: "Plane"}

    if args.close_ended:
        with open(output_path, mode="r") as infile:
            reader = csv.DictReader(infile)
            total = [0, 0, 0, 0, 0]
            correct = [0, 0, 0, 0, 0]
            for row in reader:
                total[int(row["Question Type"]) - 1] += 1
                if row["Correct"] == "1":
                    correct[int(row["Question Type"]) - 1] += 1

        for i in range(5):
            print(f"{Qustion_Type[i + 1]}: {correct[i] / total[i]:.4f}")
    else:
        with open(output_path, mode="r") as infile:
            reader = csv.DictReader(infile)
            type = {"1": [], "2": [], "3": [], "4": [], "5": []}
            scores = {"bleu": [], "rouge1": [], "meteor": [], "bert_f1": []}
            for row in reader:
                for metric in scores.keys():
                    scores[metric].append(float(row[metric]))

                type[row["Question Type"]].append(row)

        print("\nType Average scores:")
        for k, v in type.items():
            for metric in scores.keys():
                avg = sum([float(row[metric]) for row in v]) / len(v)
                print(f"{Qustion_Type[int(k)]} {metric}: {avg:.4f}")


if __name__ == "__main__":
    main()