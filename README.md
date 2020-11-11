# Image-caption-generator-using-deep-learning

This project involves the concepts of MLP's,CNN,RNN,transfer leraning,word embeddings and language modelling.
The dataset is Flicker8k dataset available on Kaggle.
Here is the link to dataset:
https://www.kaggle.com/ming666/flicker8k-dataset/notebooks

Here all the training images contains five captions related to that image.So firstly,we created a dictionary having keys as the images and values as all captions related to that image.Then we create a vocabulary which consists of only frequently occuring unique words i.e words that have occured at least 10 times all over the training data.This helps us to reduce the size of the vocabulary.
With the help of transfer learning through Resnet50 model we extract the feature vectors of all the images(training and test).
Then all the words in the vocab are assigned to a unique number so as to convert them into numerical values.
The problem is converted into a supervised learning problem in which we create a data generator object which X contains an array of image features and partial captions and Y contains the next word.Then on the basis of these features the model predicts the new word this is language modelling.
Before feeding the word indexes into RNN model they are passed through an embedding layer which converts a word into a 50 dim-vector.Here we used glove embedding.
Now three models are created one which converts a 2048 dimensional image vector into 256 dimensional vector,one for converting sentences into 256 dimensional vector and then the output of these two models are combined in another model whose final output gives the probability of every word in the vocab after which we can take the argmax so that we get the index of word having maximum probability.
