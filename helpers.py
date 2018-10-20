import os
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import matplotlib.pyplot as plt
import seaborn as sns

def count_sensitive_words(df_, word_list_, str_col='Short_Msg', companies_neg_pos=None):
    '''
    Count number of sensitive words from `word_list_` within each day of news messages of `df_`.
    word_list_ can contain strings or dictionaries with single key:value pairs, 
    where key is the outcome feature name and value is a list of strings to be matched with OR.
    There will be also interactions of negative and positive counts with company names 
    in `companies_neg_pos` calculated.
    '''
    total_msg = df_.resample('1D').size()
    
    for w in word_list_:
        if not isinstance(w, dict):
            #extract 
            df_['{}_count'.format(w)] = df_[str_col].str.count(w)
        else:
            df_['{}_count'.format(next(iter(w.keys())))] = df_[str_col].str.upper().str.count(r'|'.join(next(iter(w.values()))))
            
    # if no companies_neg_pos is given- do interactions for all companies
    if companies_neg_pos == None:
        companies_neg_pos = [c for c in word_list_ if isinstance(c, str)]
    # do actual interactions at the message level    
    for c in companies_neg_pos:
        for w in word_list_:
            if not isinstance(w, dict):
                continue
            key, value = next(iter(w.keys())), next(iter(w.values()))
            df_['{}_{}_count'.format(c,key)] = df_['{}_count'.format(c)] * df_['{}_count'.format(key)]
            
    # total number fo messages in a day
    df_out['total_msg_count'] = total_msg
    
    # rename column names
    df_.columns = [c.replace(' ', '_') for c in df_.columns]
    # get the subset, that is interesting (counts)
    count_cols = [c for c in df_.columns if '_count' in c]
    
    # output dataset grouped by day
    df_out = df_.resample('1D')[count_cols].sum()
    del df_
    return df_out


def fe(df_):
    '''
    Feature engineering on Reuters Short_Msg BoW in general,    
    but so far it gets only the fractions of message counts
    '''
    for c in df_.columns:
        if c == 'total_msg_count':
            continue
            
        df_[c.replace('_count', '_frac')] = df_[c]/df_['total_msg_count']
        
    return df_