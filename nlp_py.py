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
import matplotlib.pyplot as plt
import re
from torch.distributions import Categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,f1_score,accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import random as rd
import json
from seaborn import heatmap
import torch.nn as nn
import torch

    """
    pyperclip.copy(string)



def tweets_disaster():
    string = '''
disaster_data = pd.read_csv('/home/luchian/Downloads/tweets_disaster.csv',usecols = ['text','target'])



def proc_text(text):
    lowered = text.lower()
    found = ' '.join(re.findall(r'[A-z0-9]+[A-z0-9]',lowered))
    return found



disaster_data['text'] = disaster_data['text'].apply(proc_text)



Train_pd, Test_pd = train_test_split(disaster_data, train_size = 0.8, test_size = 0.2,stratify=disaster_data['target'])
print(len(Train_pd))
print(len(Test_pd))



tokenizer = Tokenizer(model = tokenizers.models.WordLevel(unk_token='<unk>'))
tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
trainer = tokenizers.trainers.WordLevelTrainer(vocab_size = 500_000,special_tokens = ['<pad>','<unk>'])
tokenizer.train_from_iterator(Train_pd['text'],trainer=trainer)
tokenizer.enable_padding(direction='right',pad_id = 0,pad_token='<pad>')



vocab_size = tokenizer.get_vocab_size()
print(vocab_size)


class TextDataset(Dataset):
    def __init__(self,data,tok_obj,max_len):
        self.main_data = data
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object = tok_obj)
        self.max_length = max_len
    
    def __len__(self):
        return len(self.main_data)
    
    def __getitem__(self,indx):
        X = self.tokenizer(text = self.main_data.iloc[indx,0],
                           max_length=self.max_length,
                           padding='max_length',
                           return_tensors='pt',
                           truncation=True)['input_ids']
        y = torch.tensor(self.main_data.iloc[indx,1])
        return X,y
    


Train_dataset = TextDataset(Train_pd,tok_obj=tokenizer,max_len = 30)
Test_dataset = TextDataset(Test_pd,tok_obj = tokenizer,max_len = 30)
print(len(Train_dataset))
print(len(Test_dataset))



class ClMod(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.rel = nn.ReLU()
        self.embed = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.drop = nn.Dropout1d(p = 0.08)
        self.lin1 = nn.Linear(80,128)
        self.lin2 = nn.Linear(128,128)
        self.lin3 = nn.Linear(128,2)
        
    def forward(self,x):
        y = self.embed(x).squeeze(dim = 1)
        y = y.mean(dim = 1)
        y = self.rel(self.drop(self.lin1(y)))
        y = self.rel(self.drop(self.lin2(y)))
        y = self.rel(self.lin3(y))
        return y
    

bs = 128
epoch = 20
lr = 0.001


mod = ClMod(vocab_size = vocab_size,embed_dim=80)
optimizer = torch.optim.Adam(mod.parameters(),lr = lr,betas = (0.9,0.99))
loss = nn.CrossEntropyLoss()
loader = DataLoader(dataset=Train_dataset,shuffle = True,batch_size=bs)


def train_model(model,train_loader,epoch,main_optim,main_loss,print_every = 1):
    try:
        train_losses = []
        for ep in range(epoch):
            model.train()
            epoch_losses = []
            for X,y in tqdm(train_loader,desc=f'Going through the loader on epoch #{ep+1}'):
                main_optim.zero_grad()
                y_pred = model(X).squeeze(dim = 1)
                the_loss = main_loss(y_pred,y)
                the_loss.backward()
                main_optim.step()
                epoch_losses.append(the_loss.item())
            train_losses.append(round(np.array(epoch_losses).mean().item(),5))
            if ep%print_every == 0:
                print(f'Epoch #{ep+1} | Train loss: {train_losses[-1]}',end = '\n\n')
        return train_losses
    except KeyboardInterrupt:
        return train_losses
    

res_mod = train_model(
    model = mod,
    train_loader = loader,
    main_optim = optimizer,
    epoch = epoch,
    main_loss = loss,
    print_every=2
)



@torch.no_grad()
def get_preds(model, Val):
    model.eval()
    y_true_list = []
    y_pred_list = []
    for X,y in Val:
        probs = model(X).squeeze(dim = 1).softmax(dim = 1)
        distribution = Categorical(probs)
        y_pred = distribution.sample()
        y_true_list.append(y.item())
        y_pred_list.append(y_pred.item())
    return np.array(y_true_list),np.array(y_pred_list)



y_true, y_pred = get_preds(mod,Test_dataset)



cls_rep = classification_report(y_true = y_true,y_pred = y_pred)
print(cls_rep)



loss_figure = plt.figure(figsize = (10,5),facecolor = 'skyblue')
loss_ax = loss_figure.add_subplot()
loss_ax.plot(res_mod)
'''
    pyperclip.copy(string)



def activities():
    string = """
activities_data = pd.read_csv('/home/luchian/Downloads/activities.csv',usecols = ['Text','Review-Activity'])




label_encoder = LabelEncoder()
label_encoder.fit(activities_data['Review-Activity'])



activities_data['Review-Activity'] = label_encoder.transform(activities_data['Review-Activity'])
activities_data.head(5)



Train_pd, Test_pd = train_test_split(activities_data,train_size = 0.8,test_size = 0.2,stratify = activities_data['Review-Activity'])
print(len(Train_pd))
print(len(Test_pd))



def proc_text(text):
    lowered = text.lower()
    processed = ' '.join(re.findall(r"[A-z0-9]+[A-z0-9]",lowered))
    return processed



Train_pd['Text'] = Train_pd['Text'].apply(proc_text)
Test_pd['Text'] = Test_pd['Text'].apply(proc_text)



TFIDF = TfidfVectorizer()
TFIDF.fit(Train_pd['Text'])



dim1 = TFIDF.transform(raw_documents=Train_pd['Text']).shape[1]



class ActDataset(Dataset):
    def __init__(self, data):
        self.main_data = data
    
    def __len__(self):
        return len(self.main_data)
    
    def __getitem__(self, indx):
        X,y = TFIDF.transform([self.main_data.iloc[indx,0]]), torch.tensor(self.main_data.iloc[indx,1])
        return torch.from_numpy(X.toarray()).to(dtype = torch.float32),y.to(dtype = torch.long)
    


Train_dataset = ActDataset(Train_pd)
Test_dataset = ActDataset(Test_pd)
print(len(Train_dataset))
print(len(Test_dataset))




class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rel = nn.ReLU()
        self.drop = nn.Dropout1d(p = 0.08)
        self.lin1 = nn.Linear(dim1,128)
        self.lin2 = nn.Linear(128,128)
        self.lin3 = nn.Linear(128,2)
    def forward(self,x):
        y = self.rel(self.drop(self.lin1(x)))
        y = self.rel(self.drop(self.lin2(y)))
        y = self.lin3(y)
        return y
    



bs = 128
epoch = 10
lr = 0.0001



act_model = MainModel()
train_loader = DataLoader(dataset = Train_dataset,shuffle = True,batch_size = bs)
optimizer = torch.optim.Adam(act_model.parameters(),lr = lr,betas = (0.9,0.99))
loss = nn.CrossEntropyLoss()




def train_model(model,train_loader,epoch,main_optim,main_loss,print_every = 1):
    try:
        train_losses = []
        for ep in range(epoch):
            model.train()
            epoch_losses = []
            for X,y in tqdm(train_loader,desc=f'Going through the loader on epoch #{ep+1}'):
                main_optim.zero_grad()
                y_pred = model(X).squeeze(dim = 1)
                the_loss = main_loss(y_pred,y)
                the_loss.backward()
                main_optim.step()
                epoch_losses.append(the_loss.item())
            train_losses.append(round(np.array(epoch_losses).mean().item(),5))
            if ep%print_every == 0:
                print(f'Epoch #{ep+1} | Train loss: {train_losses[-1]}',end = '\n\n')
        return train_losses
    except KeyboardInterrupt:
        return train_losses
    


res_mod = train_model(
    model = act_model,
    train_loader = train_loader,
    main_optim = optimizer,
    epoch = epoch,
    main_loss = loss,
    print_every=1
)




@torch.no_grad()
def get_preds(model, Val):
    model.eval()
    y_true_list = []
    y_pred_list = []
    for X,y in Val:
        probs = model(X).squeeze(dim = 1).softmax(dim = 1)
        distribution = Categorical(probs)
        y_pred = distribution.sample()
        y_true_list.append(y.item())
        y_pred_list.append(y_pred.item())
    return np.array(y_true_list),np.array(y_pred_list)




y_true,y_pred = get_preds(act_model,Test_dataset)




cls_rep = classification_report(y_true = y_true,y_pred = y_pred)
print(cls_rep)



loss_figure = plt.figure(figsize=(10,5),facecolor = 'skyblue')
loss_ax = loss_figure.add_subplot()
loss_ax.plot(res_mod)
"""
    pyperclip.copy(string)



def news():
    string = ''''
news_data = pd.read_csv('/home/luchian/all_data/uni_data/NLP_datasets/news.csv', usecols = ['Class Index','Title'])
print(news_data.info(),end = '\n\n')
news_data.head(5)



label_encoder = LabelEncoder().fit(news_data['Class Index'])
label_encoder



news_data['Class Index'] = label_encoder.transform(news_data['Class Index'])



Train_news,Test_news = train_test_split(news_data,train_size = 0.9,stratify=news_data['Class Index'])
print(len(Train_news))
print(len(Test_news))



def proc_text(text):
    """simple processing of a text"""
    lowered = text.lower()
    tokenized = ' '.join(re.findall(r"[A-z0-9]+[A-z0-9]",lowered))
    return tokenized



def train_tokens():
    """creates a training generator for tokenizing"""
    for row in range(len(Train_news.iloc[:,1])):
        processed = proc_text(Train_news.iloc[row,1]) 
        yield processed



def tokenizer_to_path(gen):
    """creates a doc with voc for further use with FastTokenizer from transformers"""
    tokenizer = Tokenizer(model = tokenizers.models.WordLevel(unk_token='<unk>'))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok_trainer = tokenizers.trainers.WordLevelTrainer(vocab_size = 500_000,special_tokens = ['<pad>','<unk>'])
    tokenizer.enable_padding(direction='right',pad_id = 0,pad_token='<pad>')
    tokenizer.train_from_iterator(gen,trainer=tok_trainer)
    return tokenizer



tok = tokenizer_to_path(train_tokens())
print(type(tok))



class TextDataset(Dataset):
    def __init__(self,texts,tok_obj,processing,max_len):
        self.texts = texts
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object = tok_obj)
        self.processing = processing
        self.max_length = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, indx):
        text,label = self.processing(self.texts.iloc[indx,1]),torch.tensor(self.texts.iloc[indx,0].item())
        tokenized_text = self.tokenizer(
            text,
            max_length = self.max_length,
            padding = 'max_length',
            return_tensors='pt',
            truncation=True
        )['input_ids']
        return tokenized_text.squeeze(dim = 0),label
    


News_torch_train = TextDataset(Train_news,tok,proc_text,12)
News_torch_test = TextDataset(Test_news,tok,proc_text,12)



class ClassificationModel(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_size,bi = True,bf = True):
        super().__init__() #initialization of super class
        self.embed = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.gru = nn.GRU(input_size = embed_dim,hidden_size = hidden_size,bidirectional = bi,batch_first=bf,num_layers=2,dropout=0.5)
        self.linear = nn.Linear(600,4)

    def forward(self,x):
        y = self.embed(x)
        _,y = self.gru(y)
        y = y.transpose(0,1)
        y = y.reshape(y.shape[0],-1)
        y = self.linear(y)
        return y
    


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
                print(f'Epoch #{ep+1} | Train loss: {train_losses[-1]}',end = '\n\n')
        return train_losses
    except KeyboardInterrupt:
        return train_losses
    


weights = compute_class_weight(class_weight='balanced',y = Train_news['Class Index'],classes=np.array([_ for _ in range(4)]))
weights = torch.from_numpy(weights).to(dtype = torch.float32)
weights



vocab_size = tok.get_vocab_size()
embed_dim = 300
hidden_size = 150



my_mod = ClassificationModel(vocab_size,embed_dim,hidden_size)
my_mod




#train
epoch = 20
batch_size = 46
lr = 0.0005
loader = DataLoader(dataset = News_torch_train,shuffle = True,batch_size = batch_size)
optimizer = torch.optim.Adam(my_mod.parameters(),lr = lr,betas=(0.9,0.989))
criterion = nn.CrossEntropyLoss(reduction = 'mean',weight = weights,label_smoothing=0.0001)




results = train_model(model = my_mod,
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




y_true, y_pred = get_preds(my_mod,News_torch_test)




report = classification_report(y_true = y_true,y_pred = y_pred)
print(report)




conf_matrix = confusion_matrix(y_true = y_true,y_pred = y_pred)
heatmap(conf_matrix,annot = True)




import matplotlib.pyplot as plt
some_figure = plt.figure(figsize = (10,5),facecolor = 'skyblue')
some_ax = some_figure.add_subplot()
some_ax.set_xlabel('Epoch')
some_ax.set_ylabel('Avg Train Loss')
some_ax.grid(linestyle = '--',c = 'pink',alpha = 0.32)
some_ax.plot(results,c = 'green')




class ModelInf(object):
    def __init__(self,model,TrainSet):
        self.model = model
        self.dataset = TrainSet
    
    def __call__(self,text):
        self.model.eval()
        main_text = self.dataset.processing(text)
        tokens = self.dataset.tokenizer(
                main_text,
                max_length = self.dataset.max_length,
                padding = 'max_length',
                return_tensors='pt',
                truncation=True)['input_ids']
        y_pred = self.model(tokens).softmax(dim = 1).argmax(dim = 1).item()
        return label_encoder.classes_[y_pred].item()
    


inf = ModelInf(my_mod,News_torch_train)



t1,t2 = Test_news.iloc[rd.randint(0,999),1],Test_news.iloc[rd.randint(0,999),1]
print(f'Text: {t1}\nprediction: {inf(t1)}',end = '\n\n')
print(f'Text: {t2}\nprediction: {inf(t2)}') 
    '''
    pyperclip.copy(string)



def corona():
    string = """
main_data = pd.read_csv('/home/luchian/Downloads/corona.csv', usecols=['OriginalTweet','Sentiment'])
main_data

label_encoder = LabelEncoder().fit(main_data['Sentiment'])
label_encoder.classes_


main_data['Sentiment'] = label_encoder.transform(main_data['Sentiment'])



Train,Test = train_test_split(main_data,train_size = 0.85,stratify = main_data['Sentiment'])



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
                print(f'Epoch #{ep+1} | Train loss: {train_losses[-1]}',end = '\n\n')
        return train_losses
    except KeyboardInterrupt:
        return train_losses
    


weights = compute_class_weight(class_weight='balanced',y = Train['Sentiment'],classes=np.array([_ for _ in range(5)]))
weights = torch.from_numpy(weights).to(dtype = torch.float32)



#train
epoch = 3
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


y_pred1 = model_inf('Thex exam is pretty well managed. I like it, organisers did a pretty good job')
y_pred2 = model_inf('but I hate everything about this stupid exam')


print(y_pred1)
print(y_pred2)
"""
    pyperclip.copy(string)



def tweet_cat():
    string = '''
tweet_cat_dataset = pd.read_csv('/home/luchian/all_data/uni_data/NLP_datasets/tweet_cat.csv')
tweet_cat_dataset.head(5)




label_enc = LabelEncoder()
label_enc




tweet_cat_dataset['type'] = label_enc.fit_transform(tweet_cat_dataset['type'])



Train, TestVal = train_test_split(tweet_cat_dataset, stratify = tweet_cat_dataset['type'],train_size = 0.8)
print(len(Train))
print(len(TestVal))




Test, Val = train_test_split(TestVal, stratify = TestVal['type'], train_size = 0.8)
print(len(Test))
print(len(Val))




def proc_text(text):
    """simple processing of a text"""
    lowered = text.lower()
    tokenized = ' '.join(re.findall(r"[A-z0-9]+[A-z0-9]",lowered))
    return tokenized



def train_tokens(main_train_dataset):
    for row in range(len(main_train_dataset.iloc[:,0])):
        processed = proc_text(main_train_dataset.iloc[row,0])
        yield processed



tokenizer = Tokenizer(model = tokenizers.models.WordLevel(unk_token='<unk>'))
tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
tok_trainer = tokenizers.trainers.WordLevelTrainer(vocab_size = 500_000, special_tokens = ['<pad>','<unk>'])
tokenizer.enable_padding(direction='right', pad_id = 0, pad_token='<pad>')
tokenizer.train_from_iterator(train_tokens(Train), trainer=tok_trainer)
tokenizer.save('./tok_voc')




class TextDataset(Dataset):
    def __init__(self,texts,tok_obj,processing,max_len):
        self.texts = texts
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object = tok_obj)
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
    


main_dataset = TextDataset(texts = Train,
                           tok_obj = tokenizer,
                           processing = proc_text,
                           max_len = 30)



cat_torch_train = TextDataset(Train, tokenizer, proc_text, 30)
cat_torch_test = TextDataset(Test, tokenizer, proc_text, 30)
cat_torch_val = TextDataset(Val, tokenizer, proc_text, 30)



print(len(cat_torch_train))
print(len(cat_torch_test))
print(len(cat_torch_val))



class ClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, bi = True, bf = True):
        super().__init__() #initialization of super class
        self.embed = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.gru = nn.GRU(input_size = embed_dim, hidden_size = hidden_size, bidirectional = bi, batch_first=bf, num_layers=2, dropout=0.5)
        self.linear = nn.Linear(600,4)

    def forward(self,x):
        y = self.embed(x)
        _,y = self.gru(y)
        y = y.transpose(0,1)
        y = y.reshape(y.shape[0],-1)
        y = self.linear(y)
        return y
    


def train_model(model,
                train_loader,
                epoch,
                main_optim,
                main_loss,
                patience = 3,
                print_every = 1):
    try:
        train_losses = []
        val_metric = []
        while True:
            model.train()
            epoch_losses = []
            for X,y in tqdm(train_loader, desc=f'Going through the loader'):
                main_optim.zero_grad()
                y_pred = model(X)
                the_loss = main_loss(y_pred,y)
                the_loss.backward()
                main_optim.step()
                epoch_losses.append(the_loss.item())
            y_true_val, y_pred_val = get_preds(model, cat_torch_val)
            val_metric.append(f1_score(y_true_val,y_pred_val,average = 'macro'))
            if len(val_metric) <= patience:
                pass
            else:
                checking_val = val_metric[-patience:]
                count = 0
                #cheking whether we should stop
                for ind in range(len(checking_val)-1):
                    if checking_val[ind+1] <= checking_val[ind]:
                        count += 1
                #breaking if we go down
                if count == len(checking_val)-1:
                    print('Stop training due to patience')
                    break
            train_losses.append(round(np.array(epoch_losses).mean().item(),5))
            print(f'Train loss: {train_losses[-1]}')
            print(f'Val metric: {val_metric[-1]}',end = '\n\n')
        return train_losses, val_metric
    except KeyboardInterrupt:
        return train_losses, val_metric
    


my_model = ClassificationModel(cat_torch_test.tokenizer.vocab_size, embed_dim=150, hidden_size=150)
my_model



weights = compute_class_weight(class_weight='balanced', y = lebel_enc.fit_transform(Train['type']), classes = np.array([_ for _ in range(4)]))
weights = torch.from_numpy(weights).to(dtype = torch.float32)
weights



epoch = 3
batch_size = 46
lr = 0.0005
loader = DataLoader(dataset = cat_torch_train, shuffle = True, batch_size = batch_size)
optimizer = torch.optim.Adam(my_model.parameters(),lr = lr,betas=(0.9,0.989))
criterion = nn.CrossEntropyLoss(reduction = 'mean',weight = weights,label_smoothing=0.0001)



train_model(
    model = my_model,
    train_loader = loader,
    epoch = epoch,
    main_optim = optimizer,
    main_loss = criterion,
    print_every = 1,
    patience=2
)



@torch.no_grad()
def get_preds(model, Val):
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




y_true, y_pred = get_preds(my_model,cat_torch_test)
print(y_true.shape)
print(y_pred.shape)



f1_score(y_true,y_pred,average = 'macro')



def model_inf(text,TrainDataset,label_encoder):
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
    return label_encoder.inverse_transform([y_pred]).item()



sample = Test.iloc[rd.randint(1,100),0]
print(sample)



model_inf(sample,cat_torch_train,label_enc)
'''
    pyperclip.copy(string)



def pos():
    string = '''
with open('/home/luchian/Downloads/pos.json','r',encoding = 'utf-8') as file:
    main_data = json.load(file)



Train,Test = train_test_split(main_data,train_size=0.85)
print(len(Train))
print(len(Test))



def norm_text(text):
    text = text.lower()

    what_span = re.search(r"what's",text)
    if what_span:
        what_span = what_span.span()
        text = ' '.join(text[what_span[0]:what_span[1]].split("'")) + text[what_span[1]:]

    ofcom_span = re.search(r"ofcom",text)
    if ofcom_span:
        ofcom_span = ofcom_span.span()
        text = text[:ofcom_span[0]] + text[ofcom_span[0]:ofcom_span[1]][:2] + ' ' + text[ofcom_span[0]:ofcom_span[1]][2:] + text[ofcom_span[1]:]

    norm_text = re.findall(r'[a-z0-9&][^\s]*[a-z0-9]{0,1}',text)
    return norm_text



def sent_gen():
    main_list = []
    for bundle in Train:
        normalized = norm_text(bundle['sentence'])
        for token in normalized:
            main_list.append(token)
    return main_list




def classes_gen():
    main_set = set()
    for bundle in main_data:
        for token in bundle['tags']:
            main_set.add(token)
    return list(main_set)




clsses_iter = classes_gen()
print(len(clsses_iter))
clsses_iter



sentences_iter = sent_gen()
print(len(sentences_iter))
sentences_iter[:5]



wordlevel_model_train = tokenizers.models.WordLevel(unk_token='[UNK]')
wordlevel_trainer_train = tokenizers.trainers.WordLevelTrainer(vocab_size = 500_000,show_progress = True,special_tokens = ['[PAD]','[UNK]','[SOS]','[EOS]'])
Train_tokenizer = Tokenizer(model = wordlevel_model_train)
Train_tokenizer.train_from_iterator(sentences_iter,trainer = wordlevel_trainer_train)




wordlevel_model_tags = tokenizers.models.WordLevel(unk_token='[UNK]')
wordlevel_trainer_tags = tokenizers.trainers.WordLevelTrainer(vocab_size = 500_000,show_progress = True,special_tokens = ['[PAD]','[UNK]','[SOS]','[EOS]'])
Tag_tokenizer = Tokenizer(model = wordlevel_model_tags)
Tag_tokenizer.train_from_iterator(clsses_iter,trainer = wordlevel_trainer_tags)



class SentDataset(Dataset):
    def __init__(self,the_data,transforms = None):
        self.data = the_data
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,indx):
        bundle = self.data[indx]
        X,y = norm_text(bundle['sentence'])[:],bundle['tags'][:]
        X.insert(0,'[SOS]')
        X.append('[EOS]')
        y.insert(0,'[SOS]')
        y.append('[EOS]')
        X,y = torch.tensor([Train_tokenizer.encode(token).ids[0] for token in X],dtype = torch.long),torch.tensor([Tag_tokenizer.encode(token).ids[0] for token in y],dtype = torch.long)
        if self.transforms != None:
            X,y = self.transforms(X),self.transforms(y)
        return X,y
    



class Padding():
    def __init__(self,pad = 25):
        self.pad = pad
    def __call__(self,x):
        zero_tens = torch.zeros(self.pad,dtype = x.dtype)
        for ind in range(x.shape[0]):
            zero_tens[ind] = x[ind]
        return zero_tens
    


Train_tens_dataset = SentDataset(Train,transforms=Padding())
Test_tens_dataset = SentDataset(Test,transforms=Padding())
print(len(Train_tens_dataset))
print(len(Test_tens_dataset))





train_loader_sample = DataLoader(dataset=Train_tens_dataset,shuffle = True,batch_size = 15)
test_loader_sample = DataLoader(dataset=Test_tens_dataset,shuffle = True,batch_size=15)




class ClassificationModel(nn.Module):
    def __init__(self,vocab_size,tag_size,emebed_dim,hidden_size,bi = True,bf = True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size,emebed_dim,padding_idx=0)
        self.num_cls = tag_size
        self.seq_len = 25
        self.gru = nn.GRU(input_size = emebed_dim,hidden_size = hidden_size,bidirectional = bi,batch_first=bf)
        self.lin = nn.Linear(2*hidden_size,self.seq_len*self.num_cls)

    def forward(self,x):
        y = self.embed(x)
        _,y = self.gru(y)
        y = y.transpose(0,1)
        y = y.reshape(y.shape[0],-1)
        y = self.lin(y)
        y = y.reshape(y.shape[0],self.num_cls,self.seq_len)
        return y
    



def train_model(model,train_loader,epoch,main_optim,main_loss):
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
            epoch_losses.append(round(the_loss.item(),5))
        train_losses.append(np.array(epoch_losses).mean())
        if ep%3 == 0:
            print(f'Epoch #{ep+1} | Train loss: {train_losses[-1]}',end = '\n\n')
    return train_losses




my_model = ClassificationModel(Train_tokenizer.get_vocab_size(),Tag_tokenizer.get_vocab_size(),emebed_dim=30,hidden_size=31,bi = True)
my_model


#train
epoch = 20
batch_size = 16
lr = 0.005
optimizer = torch.optim.Adam(my_model.parameters(),lr = lr,betas=(0.9,0.989))
criterion = nn.CrossEntropyLoss(reduction = 'mean')



train_loader = DataLoader(dataset=Train_tens_dataset,shuffle = True,batch_size = batch_size)


ep_losses = train_model(
    model = my_model,
    train_loader=train_loader,
    epoch=epoch,
    main_optim = optimizer,
    main_loss = criterion
)



some_figure=  plt.figure(figsize = (10,5),facecolor = 'skyblue')
some_ax = some_figure.add_subplot()
some_ax.plot(ep_losses,c = 'red')
some_ax.set_xlabel('Epoch')
some_ax.set_ylabel('Losss')
some_ax.grid(linestyle = '--',c = 'purple',alpha = 0.38)


@torch.no_grad()
def get_preds(model,loader):
    model.eval()
    predictions = []
    true_values = []
    for X,y in loader:
        y_pred = model(X).softmax(dim = 1).argmax(dim = 1)
        y_pred = y_pred.reshape(-1).tolist()
        y = y.reshape(-1).tolist()
        for y_hat,y_true in zip(y_pred,y):
            predictions.append(y_hat)
            true_values.append(y_true)
    return true_values,predictions


test_loader = DataLoader(dataset=Test_tens_dataset,shuffle = True,batch_size=1)
y_true,y_pred = get_preds(my_model,test_loader)



conf_mat = confusion_matrix(y_true=y_true,y_pred = y_pred)
heatmap(conf_mat,annot=True)



rep = classification_report(y_true = y_true,y_pred = y_pred)
print(rep)
'''
    pyperclip.copy(string)




def quotes():
    string = '''
with open('/home/luchian/Downloads/quotes.json','r',encoding = 'utf-8') as main_file:
    quotes_data = json.load(main_file)



main_set = [(the_dict['Quote'],the_dict['Category']) for the_dict in quotes_data]
print(len(main_set))
main_set


all_labels = list({tup[1] for tup in main_set})
all_labels



label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
label_encoder.classes_




def proc_text(text):
    """simple processing of a text"""
    lowered = text.lower()
    tokenized = ' '.join(re.findall(r"[A-z0-9]+[A-z0-9]",lowered))
    return tokenized




Train, Test = train_test_split(main_set,train_size = 0.9,stratify=[item[1] for item in main_set])
print(len(Train))
print(len(Test))



def train_tokens(main_train_dataset):
    for row in range(len(main_train_dataset)):
        processed = proc_text(main_train_dataset[row])
        yield processed



for_training_tokens = train_tokens([item[0] for item in Train])
for_training_tokens



tokenizer = Tokenizer(model = tokenizers.models.WordLevel(unk_token='<unk>'))
tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
tok_trainer = tokenizers.trainers.WordLevelTrainer(vocab_size = 500_000, special_tokens = ['<pad>','<unk>'])
tokenizer.enable_padding(direction='right', pad_id = 0, pad_token='<pad>')
tokenizer.train_from_iterator(for_training_tokens, trainer=tok_trainer)
tokenizer.save('./tok_voc')




class TextDataset(Dataset):
    def __init__(self,texts,tok_obj,processing,max_len,label_encoder):
        self.texts = texts
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object = tok_obj)
        self.processing = processing
        self.max_length = max_len
        self.label_enc = label_encoder
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, indx):
        text,label = self.processing(self.texts[indx][0]),torch.tensor(self.label_enc.transform([self.texts[indx][1]]).item())
        tokenized_text = self.tokenizer(
            text,
            max_length = self.max_length,
            padding = 'max_length',
            return_tensors='pt',
            truncation=True
        )['input_ids']
        return tokenized_text.squeeze(dim = 0),label
    



cat_torch_train = TextDataset(Train, tokenizer, proc_text, 30, label_encoder)
cat_torch_test = TextDataset(Test, tokenizer, proc_text, 30, label_encoder)



print(len(cat_torch_train))
print(len(cat_torch_test))




class ClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, bi = True, bf = True):
        super().__init__() #initialization of super class
        self.embed = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.gru = nn.GRU(input_size = embed_dim, hidden_size = hidden_size, bidirectional = bi, batch_first=bf, num_layers=2, dropout=0.5)
        self.linear = nn.Linear(600,len(all_labels))

    def forward(self,x):
        y = self.embed(x)
        _,y = self.gru(y)
        y = y.transpose(0,1)
        y = y.reshape(y.shape[0],-1)
        y = self.linear(y)
        return y
    



my_model = ClassificationModel(cat_torch_test.tokenizer.vocab_size, embed_dim=150, hidden_size=150)
my_model



epoch = 2
batch_size = 46
lr = 0.0005
loader = DataLoader(dataset = cat_torch_train, shuffle = True, batch_size = batch_size)
optimizer = torch.optim.Adam(my_model.parameters(),lr = lr,betas=(0.9,0.989))
criterion = nn.CrossEntropyLoss(reduction = 'mean',label_smoothing=0.0001)




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
                print(f'Epoch #{ep+1} | Train loss: {train_losses[-1]}',end = '\n\n')
        return train_losses
    except KeyboardInterrupt:
        return train_losses
    



train_model(
    model = my_model,
    train_loader = loader,
    epoch = epoch,
    main_optim = optimizer,
    main_loss = criterion,
    print_every = 1
)




@torch.no_grad()
def get_preds(model, Val):
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




y_true, y_pred = get_preds(my_model,cat_torch_test)
print(y_true.shape)
print(y_pred.shape)




accuracy_score(y_true = y_true, y_pred = y_pred)




def model_inf(text,TrainDataset,label_encoder):
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
    return label_encoder.inverse_transform([y_pred]).item()




y_pred = model_inf('hate, revenge and all the other negatigvev feelings lead to even more disaster in your life',cat_torch_train,label_encoder)
print(y_pred)
'''
    pyperclip.copy(string)



def reviews():
    string = '''
reviews_data = []

with open('/home/luchian/Downloads/reviews.json', 'r', encoding = 'utf-8') as f:
    for line in f:
        reviews_data.append(json.loads(line))



reviews_main = []
for one_dict in reviews_data:
    reviews_main.append((one_dict['summary'],one_dict['overall']))



reviews_main_frame = pd.DataFrame({'summary':np.array([item[0] for item in reviews_main]), 'overall': np.array([item[1] for item in reviews_main])})
reviews_main_frame


label_encoder = LabelEncoder()
label_encoder.fit(reviews_main_frame['overall'])
label_encoder



reviews_main_frame['overall'] = label_encoder.fit_transform(reviews_main_frame['overall'])



reviews_train,reviews_test = train_test_split(reviews_main_frame,train_size = 0.8,stratify = reviews_main_frame['overall'])
print(len(reviews_train))
print(len(reviews_test))



def proc_text(text):
    """simple processing of a text"""
    lowered = text.lower()
    tokenized = ' '.join(re.findall(r"[A-z0-9]+[A-z0-9]",lowered))
    return tokenized



def train_tokens(main_train_dataset):
    for row in range(len(main_train_dataset)):
        processed = proc_text(main_train_dataset.iloc[row,0])
        yield processed



the_train_tokens = train_tokens(reviews_train)
the_train_tokens



tokenizer = Tokenizer(model = tokenizers.models.WordLevel(unk_token='<unk>'))
tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
tok_trainer = tokenizers.trainers.WordLevelTrainer(vocab_size = 500_000, special_tokens = ['<pad>','<unk>'])
tokenizer.enable_padding(direction='right', pad_id = 0, pad_token='<pad>')
tokenizer.train_from_iterator(the_train_tokens, trainer=tok_trainer)



class TextDataset(Dataset):
    def __init__(self,texts,tok_obj,processing,max_len,label_encoder):
        self.texts = texts
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object = tok_obj)
        self.processing = processing
        self.max_length = max_len
        self.label_enc = label_encoder
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, indx):
        text, label = self.processing(self.texts.iloc[indx,0]),torch.tensor(self.texts.iloc[indx,1])
        tokenized_text = self.tokenizer(
            text,
            max_length = self.max_length,
            padding = 'max_length',
            return_tensors='pt',
            truncation=True
        )['input_ids']
        return tokenized_text.squeeze(dim = 0),label
    


train_dataset = TextDataset(
    texts = reviews_train,
    tok_obj = tokenizer,
    processing = proc_text,
    max_len = 10,
    label_encoder = label_encoder
)

test_dataset = TextDataset(
    texts = reviews_test,
    tok_obj = tokenizer,
    processing = proc_text,
    max_len = 10,
    label_encoder = label_encoder
)



class ClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, bi = True, bf = True):
        super().__init__() #initialization of super class
        self.embed = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        self.gru = nn.GRU(input_size = embed_dim, hidden_size = hidden_size, bidirectional = bi, batch_first=bf, num_layers=3, dropout=0.5)
        #arguments of linear layers have to change
        self.linear = nn.Linear(420,5)

    def forward(self,x):
        y = self.embed(x)
        _,y = self.gru(y)
        y = y.transpose(0,1)
        y = y.reshape(y.shape[0],-1)
        y = self.linear(y)
        return y
    


vocab_size = tokenizer.get_vocab_size()
embed_dim = 100
hidden_size = 70

epoch = 35
bs = 164
lr = 0.01



the_model = ClassificationModel(
    vocab_size = vocab_size,
    embed_dim = embed_dim,
    hidden_size = hidden_size
)



loader = DataLoader(dataset = train_dataset, shuffle=True, batch_size = bs)
criterion = nn.CrossEntropyLoss(reduction = 'mean')
optimizer = torch.optim.Adam(the_model.parameters(),lr = lr,betas=(0.9,0.989))



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
                print(f'Epoch #{ep+1} | Train loss: {train_losses[-1]}',end = '\n\n')
        return train_losses
    except KeyboardInterrupt:
        return train_losses
    


train_res = train_model(
    model = the_model,
    train_loader = loader,
    epoch = epoch,
    main_optim = optimizer,
    main_loss = criterion,
    print_every = 5
)



@torch.no_grad()
def get_preds(model, Val):
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




y_true, y_pred = get_preds(
    model = the_model,
    Val = test_dataset
)



print(accuracy_score(y_true = y_true,y_pred = y_pred))



def model_inf(model,text,TrainDataset,label_encoder):
    model.eval()
    tokenizer = TrainDataset.tokenizer
    processing = TrainDataset.processing
    main_text = processing(text)
    tokens = tokenizer(
            main_text,
            max_length = TrainDataset.max_length,
            padding = 'max_length',
            return_tensors='pt',
            truncation=True)['input_ids']
    y_pred = model(tokens).softmax(dim = 1).argmax(dim = 1).item()
    return label_encoder.inverse_transform([y_pred]).item()



good_text = 'the best song i ever listened to'
bad_text = 'it is so bad i want to throw up'



y_pred1 = model_inf(
    model = the_model,
    text = good_text,
    TrainDataset = train_dataset,
    label_encoder = label_encoder
)
y_pred2 = model_inf(
    model = the_model,
    text = bad_text,
    TrainDataset = train_dataset,
    label_encoder = label_encoder
)
print(y_pred1)
print(y_pred2)
'''
    pyperclip.copy(string)

