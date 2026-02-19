This is a final year project that implements an unsupervised image clustering system using pretrained convolutional neural network embeddings to group visually similar images without labelled supervision

##############################################################################
for reference purposes

used the cifar10 dataset 
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

plan to use the tiny imagenet200 dataset 
http://cs231n.stanford.edu/tiny-imagenet-200.zip

used high definition images stored in data/high-res-test to test my model from the following:
https://unsplash.com

##############################################################################

##############################################################################
notebooks used so far (files located in ./notebooks)

pca_demonstration.ipynb
for seeing how much information is retained (visually) after pca 

embeddings_extractor.ipynb
for extracting cifar_10 dataset's feature embeddings using resnet 50
##############################################################################

##############################################################################
the following folders are not tracked by git but are required in the 
folder structure

./data
this is where the image datasets and test photo collections are stored

./embeddings
this is where the extracted embeddings of the dataset for training are stored
to prevent wasting time and computational power extracting really huge 
dataset embeddings each time

./models
this is where the trained models are stored

./results
this is there the results/metrics of each trained models are stored

a function in the datamanager class (the class that makes use of these directories) will
create these directories if they dont exist
##############################################################################

##############################################################################
config.toml

this file contains frequently used information like paths for instance.
it is accessed using the config class in config_loader.py
##############################################################################

##############################################################################
data_manager.py

this file contains 2 classes

The DataManager class:

This class was created to make loading of data, embeddings and models easier
as well as keeping track of what dataset or image folder an embedding file was created from
and what embedding file a model was trained from

functions include:

##############################################################################

