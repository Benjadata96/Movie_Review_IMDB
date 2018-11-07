from Datas_IMDB import *
from CNN_IMDB import *

all_repos = ['data/train/pos','data/train/neg','data/test/pos','data/test/neg'] 
vocabulary_class = Data_Input(all_repos,1500)
x_train, x_test, x_val, y_train, y_test, y_val, embedding_matrix= vocabulary_class.main()


model_CNN = CNN('model_test', 10, 0.01) 
model_CNN.main(x_train, x_test, x_val, y_train, y_test, y_val, embedding_matrix)

<<<<<<< HEAD
testing_class = Data_Input('data/test/pos','data/test/neg')
X_test, Y_test = testing_class.concatenated_testing_input()  
    
model_CNN = CNN('model_test') 
model_CNN.training_model(X_train,X_val,Y_train,Y_val)  
model_CNN.testing_model(X_test, Y_test)
=======


>>>>>>> branchedufeu
