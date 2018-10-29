# Movie_Review_IMDB

This project aims to do sentiment analysis on some movie reviews from IMDB. 
The dataset can be found here : http://ai.stanford.edu/~amaas/data/sentiment/
The dataset is divided in positive and negative reviews, this is why we concatenate the two inputs in the code.

We use the Google Word2Vec pre-trained embedding to represent the words. Using a CNN to do the classification, each review is of dimension (1500, 300) (1500 being the max length that I decided to fix) that we can 'consider as an image' using filters of the embedding dimension (x,300).


