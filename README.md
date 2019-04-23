# diversity_mining

This project use GloVe word embeddings. Dowload from https://nlp.stanford.edu/projects/glove/. Put related and unrelated phrases to related.txt and unrelated.txt respectively and candidates phrases to cand.txt, then run embedding.py to get embeddings and store in word_embedding.json.

Run bootstrapping.py to train the model use bootstrapping and output a diversity dictionary. Users can set the threshold, training size and conditions to stop the iteration. 
