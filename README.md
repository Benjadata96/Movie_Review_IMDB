# Movie_Review_IMDB

This project aims to do sentiment analysis on some movie reviews from IMDB using Keras to implement a Deep Learning model.
The dataset can be found here : http://ai.stanford.edu/~amaas/data/sentiment/
The dataset is divided in positive and negative reviews.

I use the Glove pre-trained embedding to represent the words (disponible here : https://nlp.stanford.edu/projects/glove/). This embedding is used in the neural network as a first layer, each word has a precise index that allows the recuperation of the vector representation. Note that this embedding layer will not be updated (trained) since it is already of high quality.

Using a CNN to do the classification, each review is of dimension (1500, 100) (1500 being the max length that I decided to fix) that can 'be considered as an image' using filters of the embedding dimension (x,100).

The 'Datas_IMDB' file is to create the inputs.
The 'CNN_IMDB' file is to create the model.
The 'Script_IMDB' file is the script to run the whole process ! 
