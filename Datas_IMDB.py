#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 17:53:44 2018

@author: benjadata
"""
import numpy as np 
import os
import nltk
import codecs
import gensim

from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords 
from keras.utils import to_categorical

 
WordEmb = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',binary=True)
List_stopwords = set(stopwords.words('english')) #get a list of english stopwords

class Data_Input():
    
    def __init__(self, files_path_pos, files_path_neg):
        self.files_path_pos = files_path_pos
        self.files_path_neg = files_path_neg
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.Y_train = None
        self.Y_val = None
        self.Y_test = None
        self.validation_split = 0.1
        self.MAX_LENGTH = 1500
        
        
    def sample_input(self, file):
        X=np.zeros((self.MAX_LENGTH,300),dtype=np.float32)
        with codecs.open(file,"r","utf-8") as review:
                text = review.readline()
                tokenized_review = nltk.word_tokenize(text)
                
        if len(tokenized_review)> self.MAX_LENGTH:
            len_review = self.MAX_LENGTH
        else : 
            len_review = len(tokenized_review)
        
        for i in range(len_review):
            if tokenized_review[i] not in WordEmb.vocab or tokenized_review[i] in List_stopwords:
                pass
            else:
                X[i]=WordEmb[tokenized_review[i]] 

        return (X)
    
    
    def global_input(self, files_path):
        
        input_array = np.zeros((200,1500,300), dtype=np.float32)
        files_list = os.listdir(files_path)
        
        for i,files in enumerate(files_list[:200]):
            X_file = self.sample_input(os.path.join(files_path,files))
            input_array[i,:,:] = X_file 
        
        input_array = input_array.reshape(-1,1500,300,1)
        
        return (input_array)
        
    def create_training_input(self, files_path, is_positive):
        
        X_train = self.global_input(files_path)
        if is_positive == 1:
            Y_train = np.ones(len(X_train))
        else : 
            Y_train = np.zeros(len(X_train))
        X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, test_size = self.validation_split, random_state = 13) 
                
        return (X_train, X_val, Y_train, Y_val)
    
    def create_testing_input(self, files_path, is_positive):
        
        X_test = self.global_input(files_path)
        if is_positive == 1:
            Y_test = np.ones(len(X_test))
        else:
            Y_test = np.zeros(len(X_test))
            
        return (X_test , Y_test)
    
    def concatenated_training_input(self):
        
        X_train_pos, X_val_pos, Y_train_pos, Y_val_pos = self.create_training_input(self.files_path_pos, 1)
        X_train_neg, X_val_neg, Y_train_neg, Y_val_neg = self.create_training_input(self.files_path_neg, 0)
        print('.. training set created ..')
        self.X_train = np.concatenate([X_train_pos, X_train_neg])
        self.X_val = np.concatenate([X_val_pos, X_val_neg])
        self.Y_train = np.concatenate([Y_train_pos, Y_train_neg])
        self.Y_train = to_categorical(self.Y_train)
        self.Y_val = np.concatenate([Y_val_pos, Y_val_neg])
        self.Y_val = to_categorical(self.Y_val)
        
        self.X_train = self.X_train.astype(np.float16)
        self.X_val = self.X_val.astype(np.float16)
        self.Y_train = self.Y_train.astype(np.float16)
        self.Y_val = self.Y_val.astype(np.float16)
        
        return (self.X_train, self.X_val, self.Y_train, self.Y_val)
    
    def concatenated_testing_input(self):
        
        X_test_pos, Y_test_pos = self.create_testing_input(self.files_path_pos, 'positive')
        X_test_neg, Y_test_neg = self.create_testing_input(self.files_path_neg, 'negative')
        print('.. testing set created ..')
        self.X_test = np.concatenate([X_test_pos, X_test_neg])
        self.Y_test = np.concatenate([Y_test_pos, Y_test_neg])
        self.Y_test = to_categorical(self.Y_test)
        
        return (self.X_test, self.Y_test)
            
 





      
    

    