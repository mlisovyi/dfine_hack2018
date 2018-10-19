import os
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import seaborn as sns

def count_sensitive_words(df_, word_list_, str_col='Short_Msg'):
    '''
    Count number of sensitive words from `word_list_` within each day of news messages of `df_`.
    word_list_ can contain strings or dictionaries with single key:value pairs, 
    where key is the outcome feature name and value is a list of strings to be matched with OR.
    '''
    
    for w in word_list_:
        if not isinstance(w, dict):
            df_['{}_count'.format(w)] = df_[str_col].str.count(w)
        else:
            df_['{}_count'.format(next(iter(w.keys())))] = df_[str_col].str.count(r'|'.join(next(iter(w.values()))))
            
    count_cols = [c for c in df_.columns if '_count' in c]
    print(df_)
    
    return df_.resample('1D')[count_cols].sum()