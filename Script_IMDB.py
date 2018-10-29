#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:24:34 2018

@author: BenjaminSalem
"""

from Datas_IMDB import *
from CNN_IMDB import *

training_class = Data_Input('data/train/pos','data/train/neg')
X_train, X_val, Y_train, Y_val = training_class.concatenated_training_input()

testing_class = Data_Input('data/test/pos','data/test/neg')
X_test, Y_test = testing_class.concatenated_testing_input()  
    
model_CNN = CNN('model_test') 
model_CNN.training_model(X_train,X_val,Y_train,Y_val)  
model_CNN.testing_model(X_test, Y_test)
