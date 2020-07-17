import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import patents
from tqdm import tqdm
from configuration_bert import BertConfig
from modeling_bert import BertForSequenceClassification
from torch.utils.data import DataLoader

# Load Data
batch_size = 32
train_dataset, val_dataset = patents('./')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
valid_dataloader = DataLoader(val_dataset, batch_size=batch_size)


# Load model
config = BertConfig()
config.num_labels = 4
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', config=config)


# Training
optimizer = optim.Adam(model.parameters(), lr=2e-5, eps=1e-8)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epoches = 8
model.to(device)


def accuracy(logits, labels):
    pred_labels = F.log_softmax(logits, dim=-1).argmax(-1)
    return torch.eq(pred_labels, labels).sum().item() / len(labels)


def train():
    train_loss = 0
    train_acc = 0
    model.train()
    for ind, batch in tqdm(enumerate(train_dataloader), ncols=80):
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        loss, logits = model(
            input_ids, attention_mask=attention_mask, labels=labels)
        train_loss += loss.item()
        train_acc += accuracy(logits, labels)
        loss.backward()
        optimizer.step()
    print(f" Training loss {train_loss / len(train_dataloader)} ")
    print(f" Training accuracy {train_acc / len(train_dataloader)} ")


def valid():
    valid_loss = 0
    valid_acc = 0
    model.eval()

    for ind, batch in tqdm(enumerate(valid_dataloader), ncols=80):
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            loss, logits = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            valid_loss += loss.item()
            valid_acc += accuracy(logits, labels)

    print(f" Valid loss {valid_loss / len(valid_dataloader)} ")
    print(f" Valid accuracy {valid_acc / len(valid_dataloader)} ")


def main():
    for epoch in range(epoches):
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epoches))
        print('Training...')
        train()
        print('Valid...')
        valid()


if __name__ == '__main__':
    main()
