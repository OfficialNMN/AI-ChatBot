import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

from nlp import tokenize,bagofwords,stem
from model import NeuralNet

with open('intents.json','r') as f:
    intents=json.load(f)
allwords=[]
tags=[]
xy=[]

for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        allwords.extend(w)
        xy.append((w,tag))

ignore=['?','.','!']
allwords=[stem(w) for w in allwords if w not in ignore]
allwords=sorted(set(allwords))
tags=sorted(set(tags))

x_train=[]
y_train=[]

for (pattern_sentence,tag) in xy:
    bag=bagofwords(pattern_sentence,allwords)
    x_train.append(bag)
    label=tags.index(tag)
    y_train.append(label)

x_train=np.array(x_train)
y_train=np.array(y_train)

#Hyper Parameters
epochs=1000
batch_size=8
learning_rate=0.001
input_size=len(x_train[0])
hidden_size=8
output_size=len(tags)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.n_samples
dataset=ChatDataset()

train_loader=DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=NeuralNet(input_size,hidden_size,output_size).to(device)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device,dtype=torch.int64)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": allwords,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')





