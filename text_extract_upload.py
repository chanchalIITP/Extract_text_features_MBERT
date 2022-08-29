import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle 
from pickle import dump
import shutil
import transformers as ppb
import torch


print('GPUs Available:', tf.config.list_physical_devices('GPU'))

print(torch.cuda.is_available())

f_path = "Test.csv"

sheets_dict = pd.read_csv(f_path)
 

import transformers  as ppb 

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-multilingual-cased')

#We can use BERT but here I am using DistillBERT because BERT requires more RAM then available in the colab,but to use BERT just uncomment the next line and comment the previous line
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
#To run the model on GPU
model.cuda()
 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



def get_feature(x):

    text=pd.DataFrame(x)

    tokenized = text.iloc[:,0].apply((lambda x: tokenizer.encode(str(x), add_special_tokens=True)))

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
            k = i
    #print(max_len)

    max_len = 128
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])


    padded = padded[:,0:50]
    #padded[1000]
    #print(padded)

    #for adding paddings
    input_ids = torch.tensor(np.array(padded))

    #print(input_ids.shape)

    #to set the paddings to zero and rest to 1
    attention_mask = np.where(padded != 0, 1, 0)

    #print(attention_mask.shape)


    input_ids = (torch.tensor(padded))
    attention_mask = (torch.tensor(attention_mask))
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)


    with torch.no_grad():
        last_hidden_states_train= model(input_ids,attention_mask)

    #print(last_hidden_states_train[0].shape)
    return last_hidden_states_train[0]





text_feat={}

counter=0

for i in range(len(sheets_dict)):
    '''if counter<100000:
       counter+=1
       continue'''
    print("step : ",i)

    img_id=sheets_dict["Img"][i]
             
    sentences=[str(sheets_dict["text"][i])]
    
    embedding=get_feature(sentences)
    
    text_feat[img_id]=embedding
    counter+=1
    #if counter>=100000:
    #  break
    

print("saving data ...")
output=open("test_mbert_gpu.pkl","wb")
pickle.dump(text_feat,output)
