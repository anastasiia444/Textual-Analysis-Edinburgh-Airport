import re
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#nltk.download('wordnet')

def cleaning_text(col):
    '''
    Takes in input a df[column], it deletes special caracters, numbers and deals with contractions like don't => do not
    '''
    pattern = r'[^\w\s]'
    number_pattern = r'\d+'
    col = col.astype(str)
    col = col.apply(lambda x: re.sub(pattern, '', x))
    col = col.apply(lambda x: re.sub(number_pattern, '', x))
    col = col.apply(lambda x: contractions.fix(x))
    col = col.apply(lambda x: x.lower())
    return col

def lemming(text):
    '''
    Takes in input a text cell, it deletes stopwords, tokenizes text and does lemming
    '''
    lemmatizer = WordNetLemmatizer()
    stop_w = stopwords.words('english')
    stop_w.extend(['edinburgh', 'airport', 'Edinburgh', 'Airport'])
    
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word not in stop_w]
    lemm_tokens = [lemmatizer.lemmatize(token) for token in filtered_words]
    lemm_text = ' '.join(lemm_tokens)
    return lemm_text

def token(text):
    '''
    Takes in input a text cell, it will tokenize it
    '''
    tokens = word_tokenize(text)
    return tokens

def clean_df(df, columns, column_mapping, satisfaction_mapping):
    df_clean = df[columns].copy()
    df_clean.rename(columns = column_mapping, inplace = True)
    df_clean['Satisfaction'] = df_clean['Satisfaction'].map(satisfaction_mapping)
    
    df_clean['Feedbacks'] = df_clean['Feedbacks'].fillna("No")
    df_clean['Feedbacks_clean'] = cleaning_text(df_clean['Feedbacks'])
    df_clean['Feedbacks_lem'] = df_clean['Feedbacks_clean'].apply(lemming)
    df_clean['Feedbacks_tokens'] = df_clean['Feedbacks_lem'].apply(token)
    
    df_clean['Services_suggestions'] = df_clean['Services_suggestions'].fillna("No")
    df_clean['Suggestions_clean'] = cleaning_text(df_clean['Services_suggestions'])
    df_clean['Suggestions_lem'] = df_clean['Suggestions_clean'].apply(lemming)
    df_clean['Suggestions_tokens'] = df_clean['Suggestions_lem'].apply(token)
    
    columns_keep = ['Rates','Satisfaction',
                'Feedbacks_clean','Feedbacks_lem','Feedbacks_tokens',
                'Suggestions_clean','Suggestions_lem','Suggestions_tokens']
    
    df_clean = df_clean[columns_keep]
    return df_clean