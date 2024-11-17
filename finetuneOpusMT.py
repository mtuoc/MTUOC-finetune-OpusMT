from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
import torch
from torch.utils.data import Dataset

import yaml
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    
import sys

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_score = float('inf')
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            eval_loss = metrics.get('eval_loss')
            if eval_loss is not None:
                if eval_loss < self.best_score:
                    self.best_score = eval_loss
                    self.wait = 0
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        control.should_early_stop = True
        return control

# Step 1: Load the Moses-format parallel corpus
def load_moses_corpus(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as f_src, open(tgt_file, 'r', encoding='utf-8') as f_tgt:
        src_sentences = f_src.readlines()
        tgt_sentences = f_tgt.readlines()
    assert len(src_sentences) == len(tgt_sentences), "Source and Target files must have the same number of lines"
    return src_sentences, tgt_sentences


###config.yaml###

if len(sys.argv)>1:
    configfile=sys.argv[1]
else:
    configfile="config.yaml"

stream = open(configfile, 'r',encoding="utf-8")
configYAML=yaml.load(stream, Loader=yaml.FullLoader)

train_src_file = configYAML["train_src_file"]
train_tgt_file = configYAML["train_tgt_file"]
val_src_file = configYAML["val_src_file"]
val_tgt_file = configYAML["val_tgt_file"]
model_name = configYAML["model_name"]
max_length = configYAML["max_length"] 

output_dir = configYAML["output_dir"]
eval_strategy = configYAML["eval_strategy"]
save_strategy = configYAML["save_strategy"]
per_device_train_batch_size = configYAML["per_device_train_batch_size"]
per_device_eval_batch_size = configYAML["per_device_eval_batch_size"]
num_train_epochs = configYAML["num_train_epochs"]
weight_decay = configYAML["weight_decay"]
#save_steps = configYAML["save_steps"]
#save_total_limit = configYAML["save_total_limit"]
logging_dir = configYAML["logging_dir"]
logging_steps = configYAML["logging_steps"]
load_best_model_at_end = configYAML["load_best_model_at_end"]
# Load the best model when finished training
metric_for_best_model = configYAML["metric_for_best_model"]
# Specify the metric to use for early stopping

patience = configYAML["patience"]
output_model_name = configYAML["output_model_name"]
output_tokenizer_name = configYAML["output_tokenizer_name"]
###

# Load source and target sentences
train_src_sentences, train_tgt_sentences = load_moses_corpus(train_src_file, train_tgt_file)
val_src_sentences, val_tgt_sentences = load_moses_corpus(val_src_file, val_tgt_file)

# Step 2: Define a custom Dataset class for the Moses corpus
class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_length=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx].strip()
        tgt_text = self.tgt_texts[idx].strip()

        # Tokenize the source sentence
        model_inputs = self.tokenizer(src_text, max_length=self.max_length, truncation=True, padding="max_length")
        
        # Tokenize the target sentence (as labels)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(tgt_text, max_length=self.max_length, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

# Load the pretrained model and tokenizer


model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Step 3: Create the datasets

train_dataset = TranslationDataset(train_src_sentences, train_tgt_sentences, tokenizer, max_length=max_length)
val_dataset = TranslationDataset(val_src_sentences, val_tgt_sentences, tokenizer, max_length=max_length)

# Step 4: Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Perform evaluation at the end of each epoch
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    #save_steps=500,
    #save_total_limit=3,
    logging_dir='./logs',
    logging_steps=500,
    load_best_model_at_end=True,  # Load the best model when finished training
    metric_for_best_model="eval_loss",  # Specify the metric to use for early stopping
)

# Step 5: Define the Trainer with Custom Early Stopping Callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(patience=patience)]  # Add early stopping callback
)

# Step 6: Train the model
trainer.train()

# Step 7: Save the fine-tuned model
trainer.save_model(output_model_name)
tokenizer.save_pretrained(output_tokenizer_name)
