import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2LMHeadModel
from transformers import GPT2Config
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import argparse
from tqdm import tqdm

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
    "--max_vocab_size",
    "-m",
    help="How many tokens in the tokenizer vocabulary?",
    type=int,
)
parser.add_argument(
    "--minimum_frequency", "-f", help="Minimum frequency for merged bigrams", type=int
)

parser.add_argument("--embedding_size", "-k", help="Size of embeddding vectors (also determines inner layer size)", type=int)

parser.add_argument("--layers", "-l", help="Number of transformer layers", type=int)

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

# Tokenizer settings
max_vocab_size = args.max_vocab_size
minimum_frequency = args.minimum_frequency

# Model settings 
embedding_size = args.embedding_size
layers = args.layers

# Training settings
number_epochs = args.number_of_epochs
device_batch_size = args.device_batch_size

tokenizer = ByteLevelBPETokenizer()

dataset = load_dataset(
    "text", data_files={"train": train_dataset_file, "valid": val_dataset_file,},
)

tokenizer.train(
    files=[train_dataset_file],
    vocab_size=max_vocab_size,
    min_frequency=minimum_frequency,
)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)


tokenizer.save_model(save_folder)
del tokenizer

tokenizer = GPT2TokenizerFast.from_pretrained(save_folder, max_len=1024)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.add_special_tokens({"eos_token": "<END>"})
tokenizer.add_special_tokens({"bos_token": "<END>"})

pad_id = tokenizer(["[PAD]"]).input_ids[0][0]
output_id = tokenizer([" OUTPUT"]).input_ids[0][0]

# Setting it to ignore the loss for the proofstate and padding tokens (PyTorch CrossEntropyLoss ignores the instances where label is set to -100)
def mask_labels_padding_and_proof_state(x):
    """"
    Mask the loss from the proof state and the padding so the validation loss and early
    stopping makes more sense.
    """
    labels = []

    before_output = True
    print(x)
    
    for f in x["input_ids"]:

        if before_output:

            labels.append(-100)
        else:
            if f == pad_id:

                labels.append(-100)
            else:
                labels.append(f)

        if f == output_id:
            before_output = False
    x["labels"] = labels
    x.pop("text")

    return x




train_dataset = dataset["train"].map(
    lambda e: tokenizer(
        e["text"], truncation=True, padding="max_length", return_attention_mask=True
    ),
    batched=True,
)
val_dataset = dataset["valid"].map(
    lambda e: tokenizer(
        e["text"], truncation=True, padding="max_length", return_attention_mask=True
    ),
    batched=True,
)

train_dataset_p = []

print("Preprocessing training dataset")
for k in tqdm(range(0, len(train_dataset))):
    #print(k)
    train_dataset_p.append(mask_labels_padding_and_proof_state(train_dataset[k]))

val_dataset_p = []

print("Preprocessing validation dataset")
for k in tqdm(range(0, len(val_dataset))):
    val_dataset_p.append(mask_labels_padding_and_proof_state(val_dataset[k]))


config = GPT2Config(
    vocab_size=max_vocab_size + 3,  # there are special tokens added
    n_positions=1024,
    n_embd=embedding_size,
    n_head=12,
    n_layer=layers,
    eos_token_id=tokenizer(["<END>"]).input_ids[0][0],
    bos_token_id=tokenizer(["<END>"]).input_ids[0][0],
    pad_token_id=tokenizer(["[PAD]"]).input_ids[0][0],
)
from typing import Optional, Tuple, Union
import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss
class CustomGPT2LMHeadModel(GPT2LMHeadModel):

    #@add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    #@add_code_sample_docstrings(
    #    checkpoint=_CHECKPOINT_FOR_DOC,
    #    output_type=CausalLMOutputWithCrossAttentions,
    #    config_class=_CONFIG_FOR_DOC,
    #)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='sum')
            #print(shift_logits.shape)
            #print(shift_labels.shape)
            #print(shift_logits.view(-1, shift_logits.size(-1)).shape)
            #print(shift_labels.view(-1).shape)
            #loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = 0
            batch_counter = 0
            for i in range(shift_logits.shape[0]):
                loss += loss_fct(shift_logits[i], shift_labels[i])
                batch_counter += 1

            # loss is the average of the total neg log prob of all the tactics    
            loss = loss / batch_counter
            


        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

model = CustomGPT2LMHeadModel(config=config)

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=save_folder,
    overwrite_output_dir=True,
    num_train_epochs=number_epochs,
    per_device_train_batch_size=device_batch_size,
    save_steps=6440,
    save_total_limit=110,
    logging_steps=100,
    logging_dir=save_folder,
    eval_steps=6440,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
)

from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset_p,
    eval_dataset=val_dataset_p,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=100)],
)

trainer.train()
