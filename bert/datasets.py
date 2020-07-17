import os
import csv
import torch
import numpy as np
from torch.utils.data import TensorDataset, random_split
from tokenization_bert import BertTokenizer
from tqdm import tqdm


def _patents(path):
    ipc_dict = {'3': 0, '5': 1, '7': 2, '99': 3}
    with open(path, encoding='utf-8') as f:
        f = csv.reader(f)
        Inputabstract = []
        IpcCode = []
        for i, line in enumerate(list(f)):
            if i > 0:  # header ['', 'abstract', 'IPC_label']
                Inputabstract.append(line[0])
                IpcCode.append(line[1])
        IpcCodeLabel = [ipc_dict[i] for i in IpcCode]
        return Inputabstract, IpcCodeLabel


def patents(data_dir, test_percentage=0.2, max_length=256, pretrained_model='bert-base-uncased'):
    Inputabstract, IpcCodeLabel = _patents(os.path.join(data_dir, 'Data.csv'))
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    input_ids = []
    attention_masks = []
    print("Loading data...")
    for sent in tqdm(Inputabstract, ncols=80):
        output_dict = tokenizer.encode(sent, max_length=max_length, padding=True,
                                       truncation=True, encode_plus=True, return_tensors=True)
        input_ids.append(output_dict["input_ids"])
        attention_masks.append(output_dict["attention_mask"])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(IpcCodeLabel)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    n_test = round(len(Inputabstract) * test_percentage)
    n_train = len(Inputabstract) - n_test
    train_dataset, val_dataset = random_split(dataset, [n_train, n_test])

    return train_dataset, val_dataset


if __name__ == '__main__':
    train_dataset, val_dataset = patents('./')
