from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd, numpy as np
import math


class OneHEncode:
    '''
    Vectorize a dataset of texts of arbitrary length using BOW/TFIDF features
    and generate train/val/test splits
    '''

    def __init__(self, path, sep):
        '''
        path:       path to CSV/TSV file
        sep:        TSV/CSV file separator (e.g. '\t' or ',')

        The expected input is a TSV file with two column headers:

            - 'Document' is used to refer to the text inputs and 
            - 'Label' to the their categories (labels, classes)
        
        '''
        # Read data
        self._df = pd.read_csv(path, sep=sep, header=0).dropna()
        self._index = self._df.index
        # Vectorize labels
        self._label_enc = OneHotEncoder()
        self._labdict = None
        self._raw_labels = self._df['Label']
        self._labels = self._define_labels()
        # Vectorize input data
        self._vec_data = self._onehotvectorize(corpus_df=self._df['Document'], n_feats=512)

    def _onehotvectorize(self, corpus_df, n_feats):
        '''
        One hot encoding - transforms each sentence into a
        matrix:

        - for each input of length N, create a N x M table
        - create a K x N x M table for the K inputs
        - results are saved in class-internal variables

        '''
        self._build_char_table(corpus_df, n_feats)
        self._set_input_toks()
        return self._build_embeddings()

    def _build_char_table(self, corpus_df, n_feats):
        '''
        Read dataset, create table of M tokens for inputs 
        '''
        self._input_texts = []
        self._input_tokens = []
        self._unigrams = {}
        for _, row in corpus_df.items():
            input_text  = row
            self._input_texts.append(str(input_text).split(" "))
            for tok in str(input_text).split(" "):
                if tok not in self._unigrams:
                    self._unigrams[tok] = 1
                else:
                    self._unigrams[tok] =+ 1
        if n_feats is not None:
            usorted = dict(sorted(self._unigrams.items(), key=lambda x: x[1]))

            #print(usorted.keys())

            self._input_tokens = list(usorted.keys())[0:n_feats]
        else:
            self._input_tokens = list(self._unigrams.keys())

    def _set_input_toks(self):
        '''
        Set internal variables
    
        - lists of input tokens
        - max number of tokens
        - max length of inputs
        - input token indexes

        '''
        self._input_tokens = sorted(list(self._input_tokens))
        self._num_encoder_tokens = len(self._input_tokens)
        # We truncate sequences to mean length
        self._max_encoder_seq_length = math.floor(np.mean([len(txt) for txt in self._input_texts]))
        self._input_token_index  = dict(
            [(tok, j) for j, tok in enumerate(self._input_tokens)])
        
        #print(self._input_texts[0])
        #print(len(self._input_texts[0]))
        #print(self._max_encoder_seq_length)

    def _build_embeddings(self):
        '''
        Create one-hot char embeddings (private method)
        '''
        # we work with one-hot token vectors
        encoder_input_data  = np.zeros((len(self._input_texts), 
                                        self._max_encoder_seq_length, 
                                        self._num_encoder_tokens), dtype='float32')
        # loop over data
        for i, input_text in enumerate(self._input_texts):
            # loop
            for t, tok in enumerate(input_text):
                #print(t, tok, len(input_text))
                if (tok in self._input_token_index) and (t < self._max_encoder_seq_length):
                    encoder_input_data[i, t, self._input_token_index[tok]] = 1.
        return encoder_input_data

    def _define_labels(self):
        labs = list(set(list(self._df['Label'].values)))
        lab_dict = {}
        for lab in labs:
            lab_dict[lab] = labs.index(lab)
        self._labdict = lab_dict
        labels = self._df['Label'].apply(lambda x: [lab_dict[x]]).values.tolist()
        self._label_enc.fit(labels)
        return self._label_enc.transform(labels)
    
    def get_label_dict(self):
        return self._labdict

    def get_vectorizer(self):
        return self._vectorizer()

    def get_raw_labels(self):
        return self._raw_labels

    def get_labels(self):
        return self._labels

    def get_raw_data(self) -> pd.DataFrame:
        return self._df

    def get_input_data(self):
        return self._vec_data

    def split_data(self):
        '''
        We use 20% for test, 10% for validation and 70% for training
        '''

        X, X_te, y, y_te = train_test_split(self._vec_data,
                                            self._labels,
                                            test_size=0.2,
                                            random_state=42)
        
        X_tr, X_va, y_tr, y_va = train_test_split(X,y,
                                            test_size=0.1,
                                            random_state=42)
        
        return (X_tr, y_tr), (X_te, y_te), (X_va, y_va)