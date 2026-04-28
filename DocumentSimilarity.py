# -*- coding:utf-8 -*-

# @Time    : 2023-08-02
# @Author  : Gene Moo Lee and Xiaoke Zhang
# Disclaimer: Students, please do not redistribute this code outside the class. Thank you!


# Note: These is one wrapper to compute firm-level embedding and similarity based on the word2vec model

import pandas as pd
import numpy as np

class DocumentSimilarity:

    def __init__(self, model, gvkeys, conm, keywordslist):
        '''
        Initialize the class
        model: the word2vec model 
        gvkeys: a list/pandas series of unique firm identifiers
        conm: a list/pandas series of company names
        keywordslist: a list of keywords

        gvkeys and keywordslist should be of the same length
        '''

        assert len(gvkeys) == len(keywordslist) == len(conm), "gvkeys, conm, keywordslist should should be of the same length"

        # store the information
        self.model = model
        self.firms = list(gvkeys)
        self.conm = list(conm)
        self.keywordslist = [x.split() for x in list(keywordslist)]

        # generate document embedding
        self.document_embeddings = [self.model.wv.get_mean_vector(x) for x in self.keywordslist]

        # convert to array to facilitate computation, normalize it
        self.document_array = np.array(self.document_embeddings)
        self.document_array = self.document_array / np.linalg.norm(self.document_array, axis=1)[:, np.newaxis]

    def get_firm_embedding(self, firm):
        '''Given the firm unique identifier, return the embedding of this firm'''

        return self.document_embeddings[self.firms.index(firm)]
    
    def similarity(self, firm1, firm2):
        '''Given two firms' unique identifiers, return the similarity between the two firms'''
        firm1 = self.document_embeddings[self.firms.index(firm1)]
        firm2 = self.document_embeddings[self.firms.index(firm2)]

        return np.dot(firm1, firm2) / (np.linalg.norm(firm1) * np.linalg.norm(firm2))
    
    def most_similar(self, firm, topn = 5):
        '''Given one firm unique identifier, return the topn similar firms to it
        firm: firm unique identifier
        topn: the number of firms to return
        '''
        
        v = self.document_embeddings[self.firms.index(firm)]
        v = v / np.linalg.norm(v)

        cosine_similarities = np.dot(self.document_array, v)

        # find the index of the top n companies
        sorted_indices = np.argsort(-cosine_similarities)
        largest_n_indices = sorted_indices[:topn + 1]

        return [(self.firms[x], self.conm[x], cosine_similarities[x]) for x in largest_n_indices[1:]]