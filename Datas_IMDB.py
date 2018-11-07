import numpy as np 
import pandas as pd 
import os

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

    
class Data_Input():
    
    def __init__(self, repos_list, MAX_LENGTH, VALIDATION_SPLIT):
        self.all_repos = repos_list
        self.MAX_LENGTH = MAX_LENGTH
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.Y_train = None
        self.Y_val = None
        self.Y_test = None
        self.VALIDATION_SPLIT = VALIDATION_SPLIT
        self.embedding_matrix = None
      
    
    def create_global_dataframe(self):
        
        df = pd.DataFrame([], columns = [['Text','Output', 'Train_or_Test']])
        df['Output'] = df['Output'].astype('int64')
        
        for repo in self.all_repos:
            all_files = os.listdir(repo)    
            
            for file in all_files:                
                f = open(os.path.join(repo,file),'r')
                text = f.readline()
                
                if len(text) > self.MAX_LENGTH:
                    text = text[:self.MAX_LENGTH]
                
                if 'train' in repo:
                    if 'pos' in repo:
                        df = df.append({'Text':text,
                                        'Output' : 1,
                                        'Train_or_Test':'train'},ignore_index = True)
                    else:
                        df = df.append({'Text':text,
                                        'Output' : 0,
                                        'Train_or_Test':'train'},ignore_index = True)
                else:
                    if 'pos' in repo:
                        df = df.append({'Text':text,
                                        'Output' : 1,
                                        'Train_or_Test':'test'},ignore_index = True)
                    else:
                        df = df.append({'Text':text,
                                        'Output' : 0,
                                        'Train_or_Test':'test'},ignore_index = True)  
        print('.. global dataframe created ..')    
        return(df)
    
    
    def implementing_datas(self, df):
        
        text_list = df.Text.tolist()
        output_array = df.Output.values
        
        Token = Tokenizer()
        Token.fit_on_texts(text_list)
        vocab_size = len(Token.word_index) + 1

        encoded_docs = Token.texts_to_sequences(text_list)
        
        padded_docs = pad_sequences(encoded_docs, maxlen=self.MAX_LENGTH, padding='post')
        
        embeddings_index = dict()
        f = open('glove.6B/glove.6B.100d.txt')
        for line in f:
            	s = line.split()
            	word = s[0]
            	coefs = np.asarray(s[1:], dtype='float32')
            	embeddings_index[word] = coefs
        f.close()
        print('.. Loaded %s word vectors.' % len(embeddings_index))
        
        embedding_matrix = np.zeros((vocab_size, 100))
        for word, i in Token.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector          
        print('.. Embedding Matrix created ..')
        
        return (padded_docs, output_array, embedding_matrix)        
    
    
    def main(self):
        
        df = self.create_global_dataframe()
        padded_docs, output_array, self.embedding_matrix = self.implementing_datas(df)
        
        split = len(padded_docs)/2
        
        self.X_train = padded_docs[:split]
        self.Y_train = output_array[:split]
        self.X_test = padded_docs[split:]
        self.Y_test = output_array[split:]
        
        self.X_test, self.X_val, self.Y_test, self.Y_val = train_test_split(
                self.X_test, self.Y_test, test_size = self.VALIDATION_SPLIT, 
                random_state=13)
        
        self.Y_train = to_categorical(self.Y_train)
        self.Y_test = to_categorical(self.Y_test)
        self.Y_val = to_categorical(self.Y_val)
        
        print('.. inputs are ready ..')
        
        return (self.X_train, self.X_test, self.X_val, 
                self.Y_train, self.Y_test, self.Y_val,
                self.embedding_matrix)

            
 





      
    

    
