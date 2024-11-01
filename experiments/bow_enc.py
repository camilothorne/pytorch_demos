from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class BoWEconde:
    '''
    Vectorize a dataset of texts of arbitrary length using BOW/TFIDF features
    and generate train/val/test splits
    '''

    def __init__(self, path, colnames, sep):
        '''
        path:       path to CSV/TSV file
        sep:        TSV/CSV file separator (e.g. '\t' or ',')
        colnames:   list of column names [C_1, ..., C_n] where:

            - 'Document' is used to refer to the text inputs and 
            - 'Label' to the their categories
        
        '''
        self._df = pd.read_csv(path, sep=sep, names=colnames).dropna()
        self._vectorizer = TfidfVectorizer()
        self._vectorizer.fit(self._df.Document)
        self._vec_data = self._vectorizer.transform(self._df.Document)
        self._labdict = None
        self._raw_labels = self._df.Label
        self._labels = self._define_labels()

    def _vectorizer(self):
        return self._vectorizer

    def _define_labels(self):
        labs = list(set(list(self.df.Labels.values)))
        lab_dict = {}
        for lab in labs:
            lab_dict[lab] = labs.index(lab)
        self._labdict = lab_dict 
        return self._df.Label.map(lab_dict)
    
    def get_label_dict(self):
        return self._labdict

    def get_vectorizer(self):
        return self._vectorizer()

    def get_raw_labels(self):
        return self._raw_labels

    def get_labels(self):
        return self._labels

    def get_raw_data(self):
        return self._df

    def get_input_data(self):
        return self._vec_data

    def split_data(self):
        '''
        We use 20% for test, 10% for validation and 70% for training
        '''
        X, X_te, y, y_te = train_test_split(self._vec_data,
                                            self._labels.values,
                                            test_size=0.2,
                                            random_state=42)
        X_tr, X_va, y_tr, y_va = train_test_split(X,y,
                                            test_size=0.1,
                                            random_state=42)
        return (X_tr, y_tr), (X_te, y_te), (X_va. y_va)