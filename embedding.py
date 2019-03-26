import nltk
import codecs
from io import open
import collections
import csv
import json
import sys
import io

words = set()


def readfile(file):
    f = open(file, "r")
    for p in f:
        if p[-1] == '\n':
            p = p[:-1]
        ws = p.split()
        words.add(ws[0])
        words.add(ws[1])


readfile('cand.txt')
readfile('related.txt')
readfile('unrelated.txt')
print(len(words))
word_dict = {}
with open('../glov.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0].lower()
        if word in words:
            cof = values[1:]
            cof = [float(i) for i in cof]
            word_dict[values[0]] = cof

print(len(word_dict))

with open("word_embedding.json", 'w') as outfile:
    outfile.write(json.dumps(word_dict))


