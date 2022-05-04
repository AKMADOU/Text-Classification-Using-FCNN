import random
from data import loaddata
import preprocessing
from torch.utils.data import DataLoader
from model import FeedfowardTextClassifier
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from train import TrainManyEpochs
from utils import plot

df = loaddata.df
print(df.head())
MAX_LEN = 128 #@param [64, 256, 512, 1024] {allow-input: true}
MAX_VOCAB = 1000 #@param [1000, 5000, 10000, 100000] {allow-input: true}

dataset = preprocessing.TextDataset(df, max_vocab=MAX_VOCAB, max_len=MAX_LEN)

train_dataset, valid_dataset, test_dataset = preprocessing.split_train_valid_test(
    dataset, valid_ratio=0.05, test_ratio=0.05)
print(len(train_dataset), len(valid_dataset), len(test_dataset))
BATCH_SIZE = 528
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=preprocessing.collate)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=preprocessing.collate)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=preprocessing.collate)

HIDDEN1 = 100 #@param [10, 30, 50, 100, 200, 500] {allow-input: true}
HIDDEN2 = 50 #@param [10, 30, 50, 100, 200, 500] {allow-input: true}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bow_model = FeedfowardTextClassifier(
    vocab_size=len(dataset.token2idx),
    hidden1=HIDDEN1,
    hidden2=HIDDEN2,
    num_labels=2,
    device=device,
    batch_size=BATCH_SIZE,
)
LEARNING_RATE = 5e-4

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, bow_model.parameters()),
    lr=LEARNING_RATE,
)
scheduler = CosineAnnealingLR(optimizer, 1)
input_type='tfidf'
train_losses, valid_losses, n_epochs = TrainManyEpochs(bow_model,train_loader,input_type, optimizer,valid_loader, criterion,scheduler)
plot(train_losses, valid_losses, n_epochs)