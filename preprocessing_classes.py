import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from itertools import compress
import contractions, unicodedata, re
from nltk.stem import LancasterStemmer

class FeatureExtractor(BaseEstimator, TransformerMixin):
    '''
    Extract features one by one for a pipeline
    '''
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.column]

class ToNumeric(BaseEstimator, TransformerMixin):
    '''
    Converts features to numeric for a pipeline
    '''
    def __init__(self, column):
        self.column = column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        numeric_df = pd.DataFrame()
        numeric_df[self.column] = pd.to_numeric(X)
        return numeric_df

class TextPreprocessor(BaseEstimator, TransformerMixin):
    '''
    Preprocessing for a pandas series containing text including:
    
    1. Replacing specific characters
    2. Expanding contractions
    3. Removing non-ASCII characters
    4. Convert to lowercase
    5. Remove punctuation
    6. Stem words
    
    '''
    
    def __init__(self, replacement_dictionary=None, column_header=None):
        self.replacement_dictionary = replacement_dictionary
        self.column_header = column_header
    
    def _replace_characters(self, X, *args):
        '''
        Replaces specific characters in the columns_to_process of X based on a replacement_dictionary
        '''
        replaced_df = pd.DataFrame()
        data = X
        for key,value in self.replacement_dictionary.items():
            data = [text.replace(key,value) for text in data]
        replaced_df = data
        return replaced_df    
    
    def _expand_contractions(self, X, *args):
        '''
        Replaces contractions with the expanded form of the word (e.g. can't to cannot) in the columns_to_process of X
        '''
        replaced_df = pd.DataFrame()
        data = X
        data = [contractions.fix(text) for text in data]
        replaced_df = data
        return replaced_df
    
    def _remove_non_ascii(self, X, *args):
        '''
        Removes non-ascii characters from the text in the columns_to_process of X
        '''
        replaced_df = pd.DataFrame()
        data = X
        non_ascii = []
        for text in data:
            text_non_ascii = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            non_ascii.append(text_non_ascii)
        replaced_df = non_ascii
        return replaced_df
    
    def _to_lowercase(self, X, *args):
        '''
        Converts all characters to lowercase in the columns_to_process of X
        '''
        replaced_df = pd.DataFrame()
        data = X
        lower_case = []
        for text in data:
            text_lower = text.lower()
            lower_case.append(text_lower)
        replaced_df = lower_case
        return replaced_df

    def _remove_punctuation(self, X, *args):
        '''
        Removes punctuation from the text in the columns_to_process of X
        '''
        replaced_df = pd.DataFrame()
        data = X
        no_punct = []
        for text in data:
            text_ex_punct = re.sub(r'[^\w\s]', '', text)
            no_punct.append(text_ex_punct)
        replaced_df = no_punct
        return replaced_df
    
    def _stem_words(self, X, *args):
        '''
        Stems the text in the columns_to_process of X
        '''
        replaced_df = pd.DataFrame()
        data = X
        stemmed_data = []
        stemmer = LancasterStemmer()
        for text in data:
            stemmed_text = []
            for word in text.split(' '):
                stemmed_text.append(stemmer.stem(word))
            stemmed_text = ' '.join(stemmed_text)
            stemmed_data.append(stemmed_text)
        replaced_df = stemmed_data
        return replaced_df

    
    def transform(self, X, *args):
        '''
        Combines all preprocessing steps for X
        '''
        print('Initialising replacing characters...')
        text_data = self._replace_characters(X)
        print('Completed replacing characters')
        print('Initialising expanding contractions...')
        text_data = self._expand_contractions(text_data)
        print('Completed expanding contractions')
        print('Initialising removing non-ascii characters...')
        text_data = self._remove_non_ascii(text_data)
        print('Completed removing non-ascii characters')
        print('Initialising converting characters to lowercase...')
        text_data = self._to_lowercase(text_data)
        print('Completed converting characters to lowercase')
        print('Initialising removal of punctuation...')
        text_data = self._remove_punctuation(text_data)
        print('Completed removal of punctuation')
        print('Initialising stemming words...')
        text_data = self._stem_words(text_data)
        print('Completed stemming words')
        text_data = pd.Series(data=text_data,index=X.index,name=self.column_header)
        return text_data
    
    def fit(self, X, *args):
        return self

class Dummifier(BaseEstimator, TransformerMixin):
    '''
    Dummifies a pandas series
    
    Ensures the resulting dummified columns match the fitted data after transformation
    '''
    
    def __init__(self):
        self.dummified_columns=None

    def transform(self, X, *args):
        '''
        Dummifies X and ensures the resulting columns match self.dummified_columns (created during fitting)
        
        Drops any columns in dummified X that are not in self.dummified_columns
        Adds a zero column for any columns in  self.dummified_columns that are not in dummified X
        '''
        # Dummify specific columns of X
        dummified_data = pd.get_dummies(X,drop_first=False)
        
        # Filter out dummified columns not in self.dummified_columns
        col_in_fit = list(compress(dummified_data.columns, dummified_data.columns.isin(self.dummified_columns)))
        dummified_data = dummified_data[col_in_fit]
        
        # Add columns in self.dummified_columns that are not in dummified X
        col_not_in_fit = list(set(self.dummified_columns)-set(dummified_data.columns))
        for col in col_not_in_fit:
            dummified_data[col] = 0

        return dummified_data


    def fit(self, X, *args):
        '''
        Creates an index of dummified columns after dummification of X
        Stored as self.dummified_columns
        '''
        # Dummify specific columns of X
        dummified_data = pd.get_dummies(X,drop_first=True)
        
        # Store new columns headers as self.dummified_columns
        self.dummified_columns = dummified_data.columns
        
        return self