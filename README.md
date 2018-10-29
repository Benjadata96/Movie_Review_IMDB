# Movie_Review_IMDB

This project aims to do sentiment analysis on some movie reviews from IMDB. 
The dataset can be found here : http://ai.stanford.edu/~amaas/data/sentiment/
The dataset is divided in positive and negative reviews, this is why I concatenate the two inputs in the code.

I use the Google Word2Vec pre-trained embedding to represent the words (disponible here : https://code.google.com/archive/p/word2vec/). Using a CNN to do the classification, each review is of dimension (1500, 300) (1500 being the max length that I decided to fix) that can 'be considered as an image' using filters of the embedding dimension (x,300).

The 'Datas_IMDB' file is to create the inputs.
The 'CNN_IMDB' file is to create the model.
The 'Script_IMDB' file is the script to run the whole process ! 
