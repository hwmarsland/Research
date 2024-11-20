"""
Fine Tuing DistilBert Model for Research Purposes.

Code is a combination of tutorials from:
https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multiclass_classification.ipynb#scrollTo=9gLGsw-nXWEd
https://huggingface.co/docs/transformers/training#example-of-a-fully-fledged-script

Author: Harris Marsland
"""

from datasets import load_dataset
from transformers import DistilBertModel, DistilBertTokenizer, TrainingArguments, Trainer
import numpy as np
import evaluate

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ds = load_dataset("florath/coq-facts-props-proofs-gen0-v1")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

def tokenize_function(examples):
    return tokenizer(examples["fact"], padding="max_length", truncation=True)

tokenized_datasets = ds.map(tokenize_function, batched=True)

model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# training_args = TrainingArguments(output_dir="test_trainer")

# metric = evaluate.load("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
)

trainer.train()