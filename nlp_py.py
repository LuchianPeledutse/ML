import pyperclip

def get_imports():
    string = """
import numpy as np
import pandas as pd
import tokenizers
from tqdm import tqdm
from tokenizers import pre_tokenizers
from torch.utils.data import Dataset,DataLoader
import tokenizers.trainers
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
import pyperclip
import re
from torch.distributions import Categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.utils.class_weight import compute_class_weight
import random as rd
from seaborn import heatmap
import torch.nn as nn
import torch
    """
    pyperclip.copy(string)



def get_corona():
    string = """
main_data = pd.read_csv('/home/luchian/all_data/uni_data/corona.csv', usecols=['OriginalTweet','Sentiment'])
main_data


label_encoder = LabelEncoder().fit(main_data['Sentiment'])
label_encoder.classes_


main_data['Sentiment'] = label_encoder.transform(main_data['Sentiment'])


main_data.head(5)


Train,Test = train_test_split(main_data,train_size = 0.85,stratify = main_data['Sentiment'])


Train.head(5)


Test.head(5)


def proc_text(text):
    lowered = text.lower()
    tokenized = ' '.join(re.findall(r"[A-z0-9]+[A-z0-9]",lowered))
    return tokenized


def train_tokens():
    for row in range(len(Train.iloc[:,0])):
        processed = proc_text(Train.iloc[row,0])
        yield processed


tokenizer = Tokenizer(model = tokenizers.models.WordLevel(unk_token='<unk>'))
tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
tok_trainer = tokenizers.trainers.WordLevelTrainer(vocab_size = 500_000,special_tokens = ['<pad>','<unk>'])
tokenizer.enable_padding(direction='right',pad_id = 0,pad_token='<pad>')
tokenizer.train_from_iterator(train_tokens(),trainer=tok_trainer)
tokenizer.save('./tok_voc')


class TextDataset(Dataset):
    def __init__(self,texts,tok_path,processing,max_len):
        self.texts = texts
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file = tok_path)
        self.processing = processing
        self.max_length = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, indx):
        text,label = self.processing(self.texts.iloc[indx,0]),torch.tensor(self.texts.iloc[indx,1].item())
        tokenized_text = self.tokenizer(
            text,
            max_length = self.max_length,
            padding = 'max_length',
            return_tensors='pt',
            truncation=True
        )['input_ids']
        return tokenized_text.squeeze(dim = 0),label


TrainDataset = TextDataset(texts = Train,
                           tok_path='./tok_voc',
                           processing=proc_text,
                           max_len = 70)


TestDataset = TextDataset(texts = Test,
                           tok_path='./tok_voc',
                           processing=proc_text,
                           max_len = 70)


class ClassificationModel(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_size,bi = True,bf = True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.gru = nn.GRU(input_size = embed_dim,hidden_size = hidden_size,bidirectional = bi,batch_first=bf,num_layers=2,dropout=0.5)
        self.linear = nn.Linear(600,5)

    def forward(self,x):
        y = self.embed(x)
        _,y = self.gru(y)
        y = y.transpose(0,1)
        y = y.reshape(y.shape[0],-1)
        y = self.linear(y)
        return y


vocab_size = tokenizer.get_vocab_size()
embedding_dim = 300
my_model = ClassificationModel(vocab_size=vocab_size,
                               embed_dim = embedding_dim,
                               hidden_size=150)


def train_model(model,train_loader,epoch,main_optim,main_loss,print_every = 1):
    try:
        train_losses = []
        for ep in range(epoch):
            model.train()
            epoch_losses = []
            for X,y in tqdm(train_loader,desc=f'Going through the loader on epoch #{ep+1}'):
                main_optim.zero_grad()
                y_pred = model(X)
                the_loss = main_loss(y_pred,y)
                the_loss.backward()
                main_optim.step()
                epoch_losses.append(the_loss.item())
            train_losses.append(round(np.array(epoch_losses).mean().item(),5))
            if ep%print_every == 0:
                print(f'Epoch #{ep+1} | Train loss: {train_losses[-1]}',end = '\\n\\n')
        return train_losses
    except KeyboardInterrupt:
        return train_losses


weights = compute_class_weight(class_weight='balanced',y = Train['Sentiment'],classes=np.array([_ for _ in range(5)]))
weights = torch.from_numpy(weights).to(dtype = torch.float32)
weights


#train
epoch = 25
batch_size = 54
lr = 0.0005
loader = DataLoader(dataset = TrainDataset,shuffle = True,batch_size = batch_size)
optimizer = torch.optim.Adam(my_model.parameters(),lr = lr,betas=(0.9,0.989))
criterion = nn.CrossEntropyLoss(reduction = 'mean',weight = weights,label_smoothing=0.001)


results = train_model(model = my_model,
                      train_loader = loader,
                      epoch = epoch,
                      main_optim = optimizer,
                      main_loss = criterion
                      )


@torch.no_grad()
def get_preds(model,Val):
    model.eval()
    the_loader = DataLoader(dataset = Val, shuffle = False, batch_size = 1)
    y_true_list = []
    y_pred_list = []
    for X,y in the_loader:
        probs = model(X).softmax(dim = 1)
        distribution = Categorical(probs)
        y_pred = distribution.sample()
        y_true_list.append(y.item())
        y_pred_list.append(y_pred.item())
    return np.array(y_true_list),np.array(y_pred_list)


y_true, y_pred = get_preds(my_model,TestDataset)


report = classification_report(y_true = y_true,y_pred = y_pred)
print(report)


conf_matrix = confusion_matrix(y_true = y_true,y_pred = y_pred)
heatmap(conf_matrix,annot = True)


label_encoder.classes_


import matplotlib.pyplot as plt
some_figure = plt.figure(figsize = (10,5),facecolor = 'skyblue')
some_ax = some_figure.add_subplot()
some_ax.set_xlabel('Epoch')
some_ax.set_ylabel('Avg Train Loss')
some_ax.plot(results,c = 'purple')


def model_inf(text):
    my_model.eval()
    tokenizer = TrainDataset.tokenizer
    processing = TrainDataset.processing
    main_text = processing(text)
    tokens = tokenizer(
            main_text,
            max_length = TrainDataset.max_length,
            padding = 'max_length',
            return_tensors='pt',
            truncation=True)['input_ids']
    y_pred = my_model(tokens).softmax(dim = 1).argmax(dim = 1).item()
    return label_encoder.classes_[y_pred]


model_inf('Thex exam is pretty well managed. I like it, organisers did a pretty good job')


model_inf('but I hate everything about this stupid exam')

    """
    pyperclip.copy(string)


