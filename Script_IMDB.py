from Datas_IMDB import *
from CNN_IMDB import *

all_repos = ['data/train/pos','data/train/neg','data/test/pos','data/test/neg'] 
vocabulary_class = Data_Input(all_repos,1500, 0.35)
x_train, x_test, x_val, y_train, y_test, y_val, embedding_matrix= vocabulary_class.main()


model_CNN = CNN('model_test', 10, 0.01, 'best_weights_save.hdf5') 
model_CNN.main(x_train, x_test, x_val, y_train, y_test, y_val, embedding_matrix)

