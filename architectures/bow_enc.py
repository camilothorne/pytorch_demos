from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd, numpy as np


class BoWEconde:
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
        self._vectorizer = TfidfVectorizer(max_features=512)
        self._vectorizer.fit(self._df['Document'])
        self._label_enc = OneHotEncoder()
        self._vec_data = self._vectorizer.transform(self._df['Document'])
        self._labdict = None
        self._raw_labels = self._df['Label']
        self._labels = self._define_labels()
        self._indices = np.arange(self._df.shape[0])

    def _vectorizer(self):
        return self._vectorizer

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
        return self._vectorizer

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
        X, X_te, l, y_te , _, i_te = train_test_split(self._vec_data,
                                            self._labels,
                                            self._indices,
                                            test_size=0.2,
                                            random_state=42)
        X_tr, X_va, y_tr, y_va = train_test_split(X,l,
                                            test_size=0.1,
                                            random_state=42)
        return (X_tr, y_tr), (X_te, i_te, y_te), (X_va, y_va)