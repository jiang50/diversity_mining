import torch
import json

with open("word_embedding.json", 'r') as f:
    word_dict = json.load(f)


print(len(word_dict['diverse']))
print(word_dict['workplace'])
