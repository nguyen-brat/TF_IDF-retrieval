import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import os
from numpy.linalg import norm
import argparse

def transform_input(text):
    text = text.split()
    vectorize = np.zeros(shape = (1, len(vocabulary)))
    idx = np.array([vocabulary[word] for word in text if word in vocabulary.keys()])
    vectorize[0][idx] = tfidf_transform.idf_[idx]
    return vectorize

def find_max(text, k):
    vectorize = transform_input(text)
    similar = np.matmul(vectorize, np.transpose(transform_output.toarray()))/(norm(vectorize)*norm(transform_output.toarray(), axis = 1))
    top_k = np.argpartition(similar.reshape(len(similar[0])), k)[-k:]
    return top_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str , help="input sentence need to ask")
    parser.add_argument("--k", type=int, default=3, help="top k output")
    args = parser.parse_args()
    data_path = glob.glob(os.getcwd() + "\dataset\Data\*\*")
    tfidf_transform = TfidfVectorizer(input = 'filename')
    transform_output = tfidf_transform.fit_transform(data_path)
    vocabulary = dict(sorted(tfidf_transform.vocabulary_.items(), key=lambda x: x[0]))
    if (args.input):
        best_matchs = [data_path[idx] for idx in find_max(args.input, args.k)]
        for match in best_matchs:
            with open(match, "r", encoding="utf-8") as f:
                print(f.read())
                print("--------------------------------------")
    else:
        print("You need to input sentece type \"-h or --h to know more\"")
    