import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from tqdm import tqdm
import seaborn as sns
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm

def sentiment_score(col):
    nlp = spacy.load('en_core_web_sm')
    analyzer = SentimentIntensityAnalyzer()
    col_score = col.apply(lambda x: analyzer.polarity_scores(x))
    col_score_df = pd.DataFrame(col_score.tolist(), columns=['neg', 'neu', 'pos','compound'])
    return col_score_df

def scoring_vader(df,text):
    scores = sentiment_score(df[text])
    scores = scores.merge(df[['Satisfaction']].reset_index(drop=True), how='left', left_index=True, right_index=True)
    scores = scores.merge(df[['Rates']].reset_index(drop=True), how='left', left_index=True, right_index=True)
    return(scores)

def plot_sentiment(target, df, title = None, roberta = False):
    '''
    Target is 'Rates' or 'Satisfaction'
    '''
    if roberta == True:
        column = ['roberta_pos','roberta_neu','roberta_neg']
    else:    
        column = ['pos','neu','neg']
    
    labels = ['Positive','Neutral','Negative']
    target_column = target

    fig, axs = plt.subplots(1,3,figsize = (13,3))
    fig.suptitle(f"Sentiment Analysis - Target variable: '{target}' (Dataset: {title})", fontsize = 10, y=1.05) 
    plt.subplots_adjust(hspace = 0.9)

    for i in range(3):
        sns.barplot(data=df, x=target, y=column[i], ax = axs[i], errorbar = None)
        axs[i].set_title(labels[i], fontsize = 8)

    for ax in axs:
        ax.margins(y=0.1)
        ax.margins(x=0.03)
        ax.set_ylabel('')
        ax.set_xlabel(ax.get_xlabel(),fontsize=8)
        ax.tick_params(axis='both', labelsize = 8)

        total = 0
        for j in ax.patches:
                total = total + j.get_height()
        
        for p in ax.patches:
            bar_height = p.get_height()
            percentage = 100 * bar_height / total
            ax.annotate(f'{percentage:.1f}%', 
                    (p.get_x() + p.get_width() / 2., bar_height),
                    ha='center', va='bottom',
                    fontsize=7.5, color='black')

def heatmap(df,title):
    correlation_matrix = df.corr(method = "spearman")
    sns.set_theme(style="white")
    plt.figure(figsize = (5,4))
    heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Spearman Correlation"})
    heatmap.set_title(f"{title}", fontsize = 10)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize = 10)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90, fontsize = 9)
    plt.show()


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenize = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_robert(text):
    encoded_text = tokenize(text, return_tensors ='pt') 
    output = model(**encoded_text) 
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
    'roberta_neg': round(scores[0].item(), 8),
    'roberta_neu': round(scores[1].item(), 8),
    'roberta_pos': round(scores[2].item(), 8)}
    return scores_dict

def scoring_roberta(df,column):   
    res = {}
    for i, row in tqdm(df.iterrows(), total = len(df)):
        try:
            text = row[column]
            id = row.name
            roberta_result = polarity_scores_robert(text)
            both = {**roberta_result}
            res[id] = both
        except RuntimeError:
            print(f'Broke for id {id}') #In case tokens over the limit of the model
    '''
    scores = pd.DataFrame(res).T.reset_index(drop=True).merge(df[['Satisfaction']].reset_index(drop=True), 
                 how='left', left_index=True, right_index=True).merge(df[['Rates']].reset_index(drop=True),
                 how='left', left_index=True, right_index=True)
    '''
    scores = (pd.DataFrame(res).T.reset_index(drop=True).merge(df.reset_index(drop=True), how='left', left_index=True, right_index=True))

    return scores