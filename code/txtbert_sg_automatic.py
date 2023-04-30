import math
import torch
import random
import statistics
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from transformers import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def get_sgraphs(cases_tags, s_graphs):
    cases_sg = {}
    for m in cases_tags:
        if m.replace(".png", "") in list(s_graphs.keys()):
            cases_sg[m] = s_graphs[m.replace(".png", "")]
        else:
            cases_sg[m] = ""
    return cases_sg


def encode_binary(gold):
    encoded_targets = []
    for g in gold:
        if g == "offensive":
            y = np.array([1])
        else:
            y = np.array([0])
        encoded_targets.append(y)
    return encoded_targets


def get_weighting(cases_tags: list) -> torch.Tensor:
    """
    Gets a weighting tensor for the loss function (e.g. BCEWithLogitsLoss).
    The weight is calculated by nonoffensive_labels / offensive labels.

    Parameters
    ----------
    cases_tags: list
      The tags for the cases (either "offensive" or "non-offensiv").
    Returns
    -------
    torch.Tensor:
      A tensor containing a number for the weighting.
    """
    # calculate weights for loss criterion to care for the imbalance of the training data.
    total_samples = len(cases_tags)

    # corresponds to "1"
    offensive_labels = sum(1 if val == "offensive" else 0 for val in cases_tags)
    # corresponds to "0"
    nonoffensive_labels = sum(1 if val != "offensive" else 0 for val in cases_tags)
    assert offensive_labels + nonoffensive_labels == total_samples

    # see https://discuss.pytorch.org/t/about-bcewithlogitslosss-pos-weights/22567/2
    return torch.Tensor([nonoffensive_labels / offensive_labels]).to(device)


# Training method
def training():
    # Set to train mode
    model.train()
    total_loss, total_accuracy = 0, 0
    # Iterate through the training batches
    for batch in tqdm(train_dataloader, desc="Iteration"):
        # Push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, type_id, labels = batch
        # Clear gradients
        model.zero_grad()
        # Get model outputs
        outputs = model(sent_id, attention_mask=mask, token_type_ids=type_id)
        # Get loss
        loss = criterion(outputs.logits, labels)
        # Add to the total loss
        total_loss = total_loss + loss
        # Backward pass to calculate the gradients
        loss.backward()
        # Update parameters
        optimizer.step()
    # Compute the training loss of the epoch
    epoch_loss = total_loss / len(train_dataloader)

    return epoch_loss

# Evaluation method
def evaluate():
    print("\nEvaluating...")
    # Set to eval mode
    model.eval()
    total_loss, total_accuracy = 0, 0
    # Iterate through the validation batches
    for batch in val_dataloader:
        # Push the batch to gpu
        batch = [t.to(device) for t in batch]
        sent_id, mask, type_id, labels = batch
        # Deactivate autograd
        with torch.no_grad():
            # Get model outputs
            outputs = model(sent_id, attention_mask=mask, token_type_ids=type_id)
            # Get loss
            loss = criterion(outputs.logits, labels)
            total_loss = total_loss + loss

    # Compute the validation loss of the epoch
    epoch_loss = total_loss / len(val_dataloader)

    return epoch_loss


# Load and prepare data
train_path = "memegraphs/data/Training_meme_dataset.csv"
val_path = "memegraphs/data/Validation_meme_dataset.csv"

# Read data
# memes_sg_df = pd.read_csv("memegraphs_sg_kb.csv")
# memes_sg = dict(zip(memes_sg_df["filename"], memes_sg_df["scene_graph"]))
memes_sg_df = pd.read_csv("memegraphs/data/multi_off_sc_automatic.csv")
memes_sg = dict(zip(memes_sg_df["filename"], memes_sg_df["scene_graph"]))

train_data = pd.read_csv(train_path)
train_data["sentence"] = train_data["sentence"].apply(lambda x: x.lower())
train_data["image_name"] = train_data["image_name"].apply(lambda x: x.replace(".png", ""))
train_data["image_name"] = train_data["image_name"].apply(lambda x: x.replace(".jpg", ""))
train_cases_texts = dict(zip(train_data["image_name"], train_data["sentence"]))
train_cases_tags = dict(zip(train_data["image_name"], train_data["label"]))
train_cases_sg = get_sgraphs(train_cases_tags, memes_sg)

val_data = pd.read_csv(val_path)
val_data["sentence"] = val_data["sentence"].apply(lambda x: x.lower())
val_data["image_name"] = val_data["image_name"].apply(lambda x: x.replace(".png", ""))
val_data["image_name"] = val_data["image_name"].apply(lambda x: x.replace(".jpg", ""))
val_cases_texts = dict(zip(val_data["image_name"], val_data["sentence"]))
val_cases_tags = dict(zip(val_data["image_name"], val_data["label"]))
val_cases_sg = get_sgraphs(val_cases_tags, memes_sg)

# Define BERT model
bert_name = "bert-base-uncased"

# Construct a BERT tokenizer based on WordPiece
bert_tokenizer = BertTokenizerFast.from_pretrained(bert_name)

# Calculate average of training data
lens = [len(bert_tokenizer.encode(train_cases_texts[c], train_cases_sg[c])) for c in list(train_cases_texts.keys())]
mean_len = statistics.mean(lens)
print("the average length is", mean_len)
max_len = math.ceil(mean_len)
print(max_len)

# Tokenize and encode sentences in each set
x_train = bert_tokenizer.__call__(
    list(train_cases_texts.values()), list(train_cases_sg.values()),
    max_length = max_len,
    padding=True,
    truncation=True,
    return_token_type_ids = True,
    return_attention_mask = True,
    return_tensors="pt"
)
x_val = bert_tokenizer.__call__(
    list(val_cases_texts.values()), list(val_cases_sg.values()),
    max_length = max_len,
    padding=True,
    truncation=True,
    return_token_type_ids = True,
    return_attention_mask = True,
    return_tensors="pt"
)

train_targets = encode_binary(list(train_cases_tags.values()))
val_targets = encode_binary(list(val_cases_tags.values()))

train_y = torch.FloatTensor(train_targets)
val_y = torch.FloatTensor(val_targets)

batch_size = 16

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Calculate weights
weighting = get_weighting(list(train_cases_tags.values()))

# Define the loss
criterion = nn.BCEWithLogitsLoss(pos_weight=weighting)

epochs = 100
# Define the number of epochs to wait for early stopping
patience = 3

seeds = [28407, 53497, 36632, 23865, 51044, 20919, 99867, 9428, 18188, 80992,
         92357, 69448, 59241, 3929, 80762, 40255, 98970, 4242, 9459, 34028]
for k in range(20):
    seed = seeds[k]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Define the name of the model
    model_name = "txt_bert_sg_" + "automatic" + str(k + 1)

    # Create a dataloader for each set
    print("Creating dataloaders...")
    # TensorDataset: Creates a PyTorch dataset object to load data from
    train_dataset = TensorDataset(x_train['input_ids'], x_train['attention_mask'], x_train["token_type_ids"], train_y)
    # DataLoader: a Python iterable over a dataset
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    val_dataset = TensorDataset(x_val['input_ids'], x_val['attention_mask'], x_val["token_type_ids"], val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained(bert_name, num_labels=1,
                                                          output_attentions=True)

    model = model.to(device)

    # Define the optimizer and the learning rate
    optimizer = AdamW(model.parameters(), lr=2e-5)

    best_val_loss = float('inf')
    best_epoch = -1
    train_losses = []
    val_losses = []

    print("Starting training...")
    # Train the model
    for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        train_loss = training()
        val_loss = evaluate()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print("\nTraining Loss:", train_loss)
        print("Validation Loss:", val_loss)

        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            # Save the model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = model_name + ".bin"
            torch.save(model_to_save.state_dict(), output_model_file)

        # Early stopping
        if ((epoch - best_epoch) >= patience):
            print("No improvement in", patience, "epochs. Stopped training.")
            break