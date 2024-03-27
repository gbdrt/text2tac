import os
import argparse
import re
import numpy as np
import pandas as pd
import nltk
import evaluate
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import (
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import wandb

os.environ["WANDB_PROJECT"] = "graph2text"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"





parser = argparse.ArgumentParser()

parser.add_argument(
    "--train_dataset_file", "-t", help="Train dataset location", type=str
)
parser.add_argument(
    "--val_dataset_file", "-v", help="Validation dataset location", type=str
)
parser.add_argument(
    "--save_folder",
    "-s",
    help="Folder where the tokenizer, model parameters and logs get stored",
    type=str,
)
parser.add_argument(
    "--number_of_epochs", "-e", help="How many passes over the training data", type=int
)

parser.add_argument(
    "--device_batch_size",
    "-b",
    help="How many samples per device per batch (lower if memory issues)",
    type=int,
)

args = parser.parse_args()

# General settings
train_dataset_file = args.train_dataset_file
val_dataset_file = args.val_dataset_file
save_folder = args.save_folder

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

dataset = load_dataset(
    "text",
    data_files={
        "train": train_dataset_file,
        "valid": val_dataset_file,
    },
)


def parse_line(line):
    pattern = r"(.+?)\s+OUTPUT\s+(.+?)\s+<END>"
    match = re.search(pattern, line)
    return (match.group(1), match.group(2))


def preprocess_function(examples):
    """split goal/tactic, tokenize the text, and set the labels"""
    # The "inputs" are the tokenized answer:
    inputs = [parse_line(line) for line in examples["text"]]
    (goals, tactics) = zip(*inputs)

    model_inputs = tokenizer(goals, max_length=128, truncation=True)

    # The "labels" are the tokenized outputs:
    labels = tokenizer(text_target=tactics, max_length=512, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(preprocess_function, batched=True)

args = Seq2SeqTrainingArguments(
    save_folder,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    learning_rate=4e-5,
    per_device_train_batch_size=args.device_batch_size,
    per_device_eval_batch_size=args.device_batch_size,
    weight_decay=0.01,
    num_train_epochs=args.number_of_epochs,
    predict_with_generate=True,
    fp16=False,
    load_best_model_at_end=False,
    report_to="wandb",
)


nltk.download('punkt')
metric = evaluate.load("rouge")
data_collator = DataCollatorForSeq2Seq(tokenizer)


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    data = {"inputs": dataset["valid"]["text"], "labels":decoded_labels, "preds":decoded_preds}
    wandb.log({"valid": wandb.Table(dataframe=pd.DataFrame.from_dict(data))})
        
    # rougeLSum expects newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    return result


trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["valid"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
