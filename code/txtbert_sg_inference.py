import math
import torch
import argparse
import statistics
import numpy as np
import pandas as pd
import torch.nn as nn

from tqdm import tqdm
from scipy import stats
from scipy.special import expit
from transformers import AdamW
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import BertTokenizerFast, BertConfig, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


parser = argparse.ArgumentParser(description="Takes as arguments the training and the testing mode")
parser.add_argument("training", help="Training mode")
parser.add_argument("testing", help="Testing mode")


def get_sgraphs(cases_tags, s_graphs):
    cases_sg = {}
    for m in cases_tags:
        if m.replace(".png", "") in list(s_graphs.keys()) and not pd.isna(s_graphs[m.replace(".png", "")]):
            cases_sg[m] = s_graphs[m.replace(".png", "")]
        else:
            cases_sg[m] = ""
    return cases_sg


def encode_binary_test(gold):
    encoded_targets = []
    for g in gold:
        if g == "offensive":
            encoded_targets.append(1)
        else:
            encoded_targets.append(0)
    return np.array(encoded_targets)


def get_probs(model, dataloader, device):
    y_pred = []
    for case_num, batch in tqdm(enumerate(dataloader)):
        batch = [t.to(device) for t in batch]
        sent_id, mask, type_id = batch
        with torch.no_grad():
            # Get output probs from both models
            outputs = model(sent_id, attention_mask=mask, token_type_ids=type_id)
            probs = expit(outputs.logits.detach().cpu().numpy()[0])
            y_pred.append(probs)
    return np.array(y_pred)


def predict_n_save(predictions, cases_tags, set, threshold, model_name, k):
    results = {}
    for i in range(len(predictions)):
        if predictions[i][0] >= threshold:
            results[list(cases_tags.keys())[i]] = 1
        else:
            results[list(cases_tags.keys())[i]] = 0
    # Save results
    res_df = pd.DataFrame(zip(list(results.keys()), list(results.values()), predictions),
                      columns=["image_name", "prediction", "prob"])
    res_df.to_csv(model_name + "_" + set + "_results" + str(k+1) + ".tsv", sep="\t", index=False)

    return results

def sg_inference(train_mode="manual", test_mode="manual"):
    dev_path = "memegraphs/data/Validation_meme_dataset.csv"
    test_path = "memegraphs/data/Testing_meme_dataset.csv"

    # Read data
    if train_mode != test_mode:
        memes_sg_df = pd.read_csv("memegraphs/data/multi_off_sc_auto_cor.csv")
        memes_sg_manual_df = pd.read_csv("memegraphs/data/multi_off_sc_new.csv")
        memes_sg_manual = dict(zip(memes_sg_manual_df["filename"], memes_sg_manual_df["scene_graph"]))
    elif train_mode == "manual":
        memes_sg_df = pd.read_csv("memegraphs/data/multi_off_sc_new.csv")
    else:
        memes_sg_df = pd.read_csv("memegraphs/data/multi_off_sc_automatic.csv")

    memes_sg = dict(zip(memes_sg_df["filename"], memes_sg_df["scene_graph"]))

    test_data = pd.read_csv(test_path)
    test_data["sentence"] = test_data["sentence"].apply(lambda x: x.lower())
    test_data["image_name"] = test_data["image_name"].apply(lambda x: x.replace(".png", ""))
    test_data["image_name"] = test_data["image_name"].apply(lambda x: x.replace(".jpg", ""))
    test_cases_texts = dict(zip(test_data["image_name"], test_data["sentence"]))
    test_cases_tags = dict(zip(test_data["image_name"], test_data["label"]))
    test_true = encode_binary_test(list(test_cases_tags.values()))

    dev_data = pd.read_csv(dev_path)
    dev_data["sentence"] = dev_data["sentence"].apply(lambda x: x.lower())
    dev_data["image_name"] = dev_data["image_name"].apply(lambda x: x.replace(".png", ""))
    dev_data["image_name"] = dev_data["image_name"].apply(lambda x: x.replace(".jpg", ""))
    dev_cases_texts = dict(zip(dev_data["image_name"], dev_data["sentence"]))
    dev_cases_tags = dict(zip(dev_data["image_name"], dev_data["label"]))
    dev_true = encode_binary_test(list(dev_cases_tags.values()))

    test_cases_sg = get_sgraphs(test_cases_tags, memes_sg)
    dev_cases_sg = get_sgraphs(dev_cases_tags, memes_sg)

    train_path = "memegraphs/data/Training_meme_dataset.csv"
    train_data = pd.read_csv(train_path)
    train_data["image_name"] = train_data["image_name"].apply(lambda x: x.replace(".png", ""))
    train_data["image_name"] = train_data["image_name"].apply(lambda x: x.replace(".jpg", ""))
    train_data["sentence"] = train_data["sentence"].apply(lambda x: x.lower())
    train_cases_texts = dict(zip(train_data["image_name"], train_data["sentence"]))
    if train_mode != test_mode:
        train_cases_sg = get_sgraphs(train_cases_texts, memes_sg_manual)
    else:
        train_cases_sg = get_sgraphs(train_cases_texts, memes_sg)

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

    x_test = bert_tokenizer.__call__(
            list(test_cases_texts.values()), list(test_cases_sg.values()),
            max_length=max_len,
            padding=True,
            truncation=True,
            return_token_type_ids = True,
            return_attention_mask = True,
            return_tensors = "pt"
        )

    x_dev = bert_tokenizer.__call__(
            list(dev_cases_texts.values()), list(dev_cases_sg.values()),
            max_length=max_len,
            padding=True,
            truncation=True,
            return_token_type_ids = True,
            return_attention_mask = True,
            return_tensors = "pt"
        )

    best_threshold = 0.5

    # Define the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dev_f1s = []
    dev_ps = []
    dev_rs = []
    test_f1s = []
    test_ps = []
    test_rs = []

    for k in range(20):
        # Define the name of the model
        model_name = "txt_bert_sg_" + train_mode + str(k + 1)

        dev_data = TensorDataset(x_dev['input_ids'], x_dev['attention_mask'], x_dev["token_type_ids"])
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=1)

        test_data = TensorDataset(x_test['input_ids'], x_test['attention_mask'], x_test["token_type_ids"])
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=1)

        model_state_dict = torch.load(model_name + ".bin")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1,
                                                              state_dict=model_state_dict,
                                                              output_attentions=True)
        model.to(device)
        model.eval()

        # Dev inference
        dev_predictions = get_probs(model, dev_dataloader, device)
        dev_results = predict_n_save(dev_predictions, dev_cases_tags, "dev", best_threshold, model_name, k)

        # Test inference
        test_predictions = get_probs(model, test_dataloader, device)
        test_results = predict_n_save(test_predictions, test_cases_tags, "test", best_threshold, model_name, k)

        print("************ Run ", str(k + 1), " ************\n")
        dev_f1 = round(f1_score(dev_true, list(dev_results.values()), average="binary"), 3)
        dev_p = round(precision_score(dev_true, list(dev_results.values())), 3)
        dev_r = round(recall_score(dev_true, list(dev_results.values())), 3)
        print("Scores for dev set")
        print("F1 =", dev_f1)
        # print("F1 =", f1_score(dev_true, list(dev_results.values()), average=None))
        print("p =", dev_p)
        print("r =", dev_r)

        test_f1 = round(f1_score(test_true, list(test_results.values()), average="binary"), 3)
        test_p = round(precision_score(test_true, list(test_results.values())), 3)
        test_r = round(recall_score(test_true, list(test_results.values())), 3)
        print("Scores for test set")
        print("F1 =", test_f1)
        # print("F1 =", f1_score(test_true, list(test_results.values()), average=None))
        print("p =", test_p)
        print("r =", test_r)
        print("********************************\n")

        dev_f1s.append(dev_f1)
        dev_ps.append(dev_p)
        dev_rs.append(dev_r)
        test_f1s.append(test_f1)
        test_ps.append(test_p)
        test_rs.append(test_r)


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


if __name__ == "__main__":
    args = parser.parse_args()
    training = args.training
    testing = args.testing

    sg_inference(training, testing)
