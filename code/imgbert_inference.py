import random
import math
import torch
import pickle
import statistics
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from scipy import stats
from transformers import AdamW
from scipy.special import expit
from build_imgbert import ImgBERT
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification, BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def encode_binary_test(gold):
    encoded_targets = []
    for g in gold:
        if g == "offensive":
            encoded_targets.append(1)
        else:
            encoded_targets.append(0)
    return np.array(encoded_targets)


def get_probs(model, dataloader):
    y_pred = []
    for case_num, batch in tqdm(enumerate(dataloader)):
        batch = [t.to(device) for t in batch]
        sent_id, mask, imgs = batch
        with torch.no_grad():
            # Get output probs from both models
            outputs, _ = model(sent_id, mask=mask, img_emb=imgs)
            probs = expit(outputs.detach().cpu().numpy()[0])
            y_pred.append(probs)
    return np.array(y_pred)


def predict_n_save(predictions, cases_tags, set, k):
    results = {}
    for i in range(len(predictions)):
        if predictions[i][0] >= best_threshold:
            results[list(cases_tags.keys())[i]] = 1
        else:
            results[list(cases_tags.keys())[i]] = 0
    # Save results
    res_df = pd.DataFrame(zip(list(results.keys()), list(results.values()), predictions),
                      columns=["image_name", "prediction", "prob"])
    res_df.to_csv(model_name + "_" + set + "_results" + str(k+1) + ".tsv", sep="\t", index=False)

    return results

dev_path = "memegraphs/data/Validation_meme_dataset.csv"
test_path = "memegraphs/data/Testing_meme_dataset.csv"

# Read data
dev_data = pd.read_csv(dev_path)
dev_data["sentence"] = dev_data["sentence"].apply(lambda x: x.lower())
dev_cases_texts = dict(zip(dev_data["image_name"], dev_data["sentence"]))
dev_cases_tags = dict(zip(dev_data["image_name"], dev_data["label"]))
dev_true = encode_binary_test(list(dev_cases_tags.values()))

test_data = pd.read_csv(test_path)
test_data["sentence"] = test_data["sentence"].apply(lambda x: x.lower())
test_cases_texts = dict(zip(test_data["image_name"], test_data["sentence"]))
test_cases_tags = dict(zip(test_data["image_name"], test_data["label"]))
test_true = encode_binary_test(list(test_cases_tags.values()))

train_path = "memegraphs/data/Training_meme_dataset.csv"
train_data = pd.read_csv(train_path)
train_data["sentence"] = train_data["sentence"].apply(lambda x: x.lower())
train_cases_texts = dict(zip(train_data["image_name"], train_data["sentence"]))

img_emb_path = "memegraphs/data/image_embeddings-densenet161.bin"
with open(img_emb_path, "rb") as i:
    all_images_embs = pickle.load(i)

test_images_embs = {k: all_images_embs[k] for k in list(test_cases_texts.keys())}
dev_images_embs = {k: all_images_embs[k] for k in list(dev_cases_texts.keys())}
# test_images = torch.tensor(np.array(list(test_images_embs.values())))

# Define BERT model
bert_name = "bert-base-uncased"
# Construct a BERT tokenizer based on WordPiece
bert_tokenizer = BertTokenizerFast.from_pretrained(bert_name)

# Calculate average length of train sentences
lens = [len(bert_tokenizer.encode(s)) for s in list(train_cases_texts.values())]
mean_len = statistics.mean(lens)
print("the average length is", mean_len)
# And set it as the maximum length
max_len = math.ceil(mean_len)

x_test = bert_tokenizer.__call__(
        list(test_cases_texts.values()),
        max_length=max_len,
        padding=True,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

x_dev = bert_tokenizer.__call__(
    list(dev_cases_texts.values()),
    max_length=max_len,
    padding=True,
    truncation=True,
    return_token_type_ids=True,
    return_attention_mask=True,
    return_tensors="pt"
)

test_images = torch.tensor(np.array(list(test_images_embs.values())))
dev_images = torch.tensor(np.array(list(dev_images_embs.values())))

best_threshold = 0.5

# Define the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

example_embedding = np.array(list(all_images_embs.values())[0])
image_embeddings_size = len(example_embedding.reshape(-1))

# Inference
dev_f1s = []
dev_ps = []
dev_rs = []
test_f1s = []
test_ps = []
test_rs = []

# Weighted
w_dev_f1s = []
w_dev_ps = []
w_dev_rs = []
w_test_f1s = []
w_test_ps = []
w_test_rs = []
####
for k in range(20):
    # Define the name of the model
    model_name = "img_bert" + str(k + 1)

    dev_data = TensorDataset(x_dev['input_ids'], x_dev['attention_mask'], dev_images)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=1)

    test_data = TensorDataset(x_test['input_ids'], x_test['attention_mask'], test_images)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=1)

    print("Loading models...")
    bert = BertModel.from_pretrained(bert_name, output_attentions=True)

    # Load model for hate
    model_state_dict = torch.load(model_name + ".bin")
    model = ImgBERT(bert, 1, image_embeddings_size)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    # Dev inference
    dev_predictions = get_probs(model, dev_dataloader)
    dev_results = predict_n_save(dev_predictions, dev_cases_tags, "dev", k)

    # Test inference
    test_predictions = get_probs(model, test_dataloader)
    test_results = predict_n_save(test_predictions, test_cases_tags, "test", k)

    print("************ Run ", str(k + 1), " ************\n")
    dev_f1 = round(f1_score(dev_true, list(dev_results.values()), average="binary"), 3)
    dev_p = round(precision_score(dev_true, list(dev_results.values())), 3)
    dev_r = round(recall_score(dev_true, list(dev_results.values())), 3)
    print("Scores for dev set")
    print("F1 =", dev_f1)
    print("F1 =", f1_score(dev_true, list(dev_results.values()), average=None))
    print("p =", dev_p)
    print("r =", dev_r)

    test_f1 = round(f1_score(test_true, list(test_results.values()), average="binary"), 3)
    test_p = round(precision_score(test_true, list(test_results.values())), 3)
    test_r = round(recall_score(test_true, list(test_results.values())), 3)
    print("Scores for test set")
    print("F1 =", test_f1)
    print("F1 =", f1_score(test_true, list(test_results.values()), average=None))
    print("p =", test_p)
    print("r =", test_r)
    print("********************************\n")

    # Weighted
    w_dev_f1 = round(f1_score(dev_true, list(dev_results.values()), average="weighted"), 3)
    w_dev_p = round(precision_score(dev_true, list(dev_results.values()), average="weighted"), 3)
    w_dev_r = round(recall_score(dev_true, list(dev_results.values()), average="weighted"), 3)

    w_test_f1 = round(f1_score(test_true, list(test_results.values()), average="weighted"), 3)
    w_test_p = round(precision_score(test_true, list(test_results.values()), average="weighted"), 3)
    w_test_r = round(recall_score(test_true, list(test_results.values()), average="weighted"), 3)
    ######

    dev_f1s.append(dev_f1)
    dev_ps.append(dev_p)
    dev_rs.append(dev_r)
    test_f1s.append(test_f1)
    test_ps.append(test_p)
    test_rs.append(test_r)

    # Weighted
    w_dev_f1s.append(w_dev_f1)
    w_dev_ps.append(w_dev_p)
    w_dev_rs.append(w_dev_r)
    w_test_f1s.append(w_test_f1)
    w_test_ps.append(w_test_p)
    w_test_rs.append(w_test_r)
    #####

best_f1 = np.amax(np.array(dev_f1s))
best_run = np.argmax(np.array(dev_f1s))
print("Best F1 score for dev set is = ", best_f1, " from run ", str(best_run + 1))
print("Test scores:")
print("p = ", test_ps[best_run])
print("r = ", test_rs[best_run])
print("F1 = ", test_f1s[best_run])

print("\nMean scores for test and error of mean:")
print("p = ", round(np.mean(test_ps), 3), "+-", round(stats.sem(test_ps), 3))
print("r = ", round(np.mean(test_rs), 3), "+-", round(stats.sem(test_rs), 3))
print("F1 = ", round(np.mean(test_f1s), 3), "+-", round(stats.sem(test_f1s), 3))

# Weighted
w_best_f1 = np.amax(np.array(w_dev_f1s))
w_best_run = np.argmax(np.array(w_dev_f1s))
print("Best weighted F1 score for dev set is = ", w_best_f1, " from run ", str(w_best_run + 1))
print("Test scores:")
print("p = ", w_test_ps[w_best_run])
print("r = ", w_test_rs[w_best_run])
print("F1 = ", w_test_f1s[w_best_run])

print("\nMean scores for test and error of mean:")
print("p = ", round(np.mean(w_test_ps), 3), "+-", round(stats.sem(w_test_ps), 3))
print("r = ", round(np.mean(w_test_rs), 3), "+-", round(stats.sem(w_test_rs), 3))
print("F1 = ", round(np.mean(w_test_f1s), 3), "+-", round(stats.sem(w_test_f1s), 3))
#####