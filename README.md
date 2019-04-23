# diversity_mining

This project aims to create an exhaustive diversity dictionary with bi-gram diversity related phrases by developing a bootstrapping learning system with neural network. Documents we used are from companies' diveristy webpage and CEO letters to shareholders, which are in letters_raw and diversity_data.

This project uses GloVe word embeddings. Dowload from https://nlp.stanford.edu/projects/glove/. Put related and unrelated phrases to related.txt and unrelated.txt respectively and candidates phrases to cand.txt, then run embedding.py to get embeddings and store them in word_embedding.json.

Run bootstrapping.py to train the model use bootstrapping and output a diversity dictionary. Users can set the threshold, training size and conditions to stop the iteration. 
