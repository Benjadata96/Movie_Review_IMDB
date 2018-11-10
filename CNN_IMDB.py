import keras
import matplotlib.pyplot as plt

from keras.models import Input,Model
from keras.layers import Reshape, Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Concatenate, Embedding
#from keras.layers.normalization import BatchNormalization
#from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import Callback   

class CNN():
    
    def __init__(self, model_name, epochs, lrate):
        self.name = model_name
        self.model = None
        self.filters_size = [3,4,5]
        self.shape = (1500,)
        self.epochs = epochs
        self.lrate = lrate
        self.decay = self.lrate / self.epochs

    
    def building_model(self, embedding_matrix):
        
        input_layer = Input(shape = self.shape)
        
        embedding_layer = Embedding(embedding_matrix.shape[0], 100, weights=[embedding_matrix],
                                    input_length = 1500, trainable = True)
        embedded_input = embedding_layer(input_layer)
        embedded_input = Dropout(0.5)(embedded_input)
        
        #reshape = Reshape((1500, 100, 1))(embedded_input)
        
        conv_layers = []        
        for fsz in self.filters_size:
            conv = Conv1D(100, fsz, input_shape=self.shape, padding = 'valid', 
                          activation = 'relu')(embedded_input)
            dropout = Dropout(0.5)(conv)
            max_pool = MaxPooling1D(pool_size = 1500-fsz+1, padding='valid')(dropout)
            conv_layers.append(max_pool)
        print('.. convs ok ..')
        
        merged_layer = Concatenate(axis=1)(conv_layers)
        flattened_layer = Flatten()(merged_layer)
        flattened_layer = Dropout(0.5)(flattened_layer)
        print('.. flattened ok ..')
        
        dense_layer = Dense(128, activation='relu')(flattened_layer)
        print('.. first dense ok ..')
        dense_layer = Dropout(0.5)(dense_layer)
        print('.. second dense ok ..')
        
        final_layer = Dense(2, activation = 'softmax')(dense_layer)
        
        self.model = Model(input_layer, final_layer)
        gradient_SGD = SGD(lr=self.lrate, momentum=0.9, decay=self.decay, nesterov=False)
        
        self.model.compile(loss='binary_crossentropy', optimizer=gradient_SGD, metrics=['accuracy'])
        print ('.. model is compiled and ready to be trained ..')
        
        print(self.model.summary())
        return(self.model)
        
    def training_model(self, X_train, X_val, Y_train, Y_val, embedding_matrix):
        
        self.model = self.building_model(embedding_matrix)
        trained_model = self.model.fit(X_train, Y_train, batch_size=100, epochs=self.epochs,verbose = 1, shuffle = True, validation_data=(X_val,Y_val))
        print('.. model is trained ..')
        
        print('.. compiling accuracy and loss ..')
        training_acc = trained_model.history['acc']
        validation_acc = trained_model.history['val_acc']
        training_loss = trained_model.history['loss']
        validation_loss = trained_model.history['val_loss']
        
        print('.. plotting accuracy and loss ..')
        epochs = range(len(training_acc))
        plt.plot(epochs, training_acc, 'bo', label = 'Training Accuracy')
        plt.plot(epochs, validation_acc, 'b', label = 'Validation Acurracy')
        plt.title('Training and Validation Acurracy')
        plt.legend()
        plt.figure()
        
        plt.plot(epochs, training_loss, 'bo', label = 'Training Loss')
        plt.plot(epochs, validation_loss, 'b', label = 'Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
        print ('.. accuracy and loss printed ..')
        
    def testing_model(self, X_test, Y_test):
        
        test_evaluation = self.model.evaluate(X_test, Y_test, verbose=1)
        print('Test Loss --> ', test_evaluation[0])
        print('Test Accuracy --> ',test_evaluation[1])
        
    
    def main(self, X_train, X_test, X_val, Y_train, Y_test, Y_val, embedding_matrix):
        
        self.training_model(X_train, X_val, Y_train, Y_val, embedding_matrix)
        self.testing_model(X_test, Y_test)
        
        
        
        
