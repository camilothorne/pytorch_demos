from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd, numpy as np


class OneHEncode:
    '''
    Vectorize a dataset of texts of arbitrary length using BOW/TFIDF features
    and generate train/val/test splits
    '''

    def __init__(self, path, sep):
        '''
        path:       path to CSV/TSV file
        sep:        TSV/CSV file separator (e.g. '\t' or ',')
        colnames:   list of column names [C_1, ..., C_n] where:

            - 'Document' is used to refer to the text inputs and 
            - 'Label' to the their categories
        
        '''
        self._df = pd.read_csv(path, sep=sep, header=0).dropna()
        self._label_enc = OneHotEncoder()
        self._vec_data = self._onehotvectorize(self.df.Document)
        self._labdict = None
        self._raw_labels = self._df['Label']
        self._labels = self._define_labels()

        print(self._vec_data.shape)

    def _onehotvectorize(self, df):
        '''
        One hot encoding - transforms each sentence into a
        matrix of M x V one hot vectors
        '''
        pass

    def _build_char_table(self, data_path, xrows, norm_len=False):
        '''
        Read (full) dataset, and:
        
        - create char table of M characters for inputs 
        - for each input of length N, create a N x M table
        - create a K x N x M table for the K inputs
        - results are saved in class-internal variables
        '''
        self.corpus = pd.read_csv(data_path, sep="\t", encoding='utf-8')
        for _, row in self.corpus.iterrows():
            input_text  = row[xrows[0]]
            self.input_texts.append(input_text)
            for char in input_text:
                if char not in self.input_characters:
                    self.input_characters.add(char)
        self._set_input_toks(norm_len)
        self._build_embeddings()

    def _set_input_toks(self, norm_len):
        '''
        Set internal variables
    
        - lists of input tokens
        - max number of tokens
        - max length of inputs
        - input token indexes
        - target token indexes
        '''
        self.input_characters = sorted(list(self._input_tokens))
        self.num_encoder_tokens = len(self._input_tokens)
        self.max_encoder_seq_length = max([len(txt) for txt in self._input_texts])
        self.input_token_index  = dict(
            [(char, j) for j, char in enumerate(self._input_tokens)])

    def _build_embeddings(self):
        '''
        Create one-hot char embeddings (private method)
        '''
        # we work with one-hot token vectors
        self.encoder_input_data  = np.zeros((len(self.input_texts), 
                                        self.max_encoder_seq_length, 
                                        self.num_encoder_tokens), dtype='float32')
        # loop over data
        for i, input_text in enumerate(self._input_texts):
            # loop
            for t, tok in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[tok]] = 1.

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