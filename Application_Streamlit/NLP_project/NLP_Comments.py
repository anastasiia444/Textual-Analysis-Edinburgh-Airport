#%%
import statistics

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn import svm
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import FreqDist
from string import punctuation

#%%
import streamlit as st
from transformers import pipeline
#%%
# Initialisation du résumeur (modèle BART)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()
#%%
data=pd.read_excel('/Users/silyamoussous/Desktop/ML/projet_env/customer_responses_2025.xlsx')

#%%
print(data.head())
del data["#"]
del data["Tags"]

#%%
data["How likely are you to recommend Edinburgh Airport to a friend or colleague?"]=data["How likely are you to recommend Edinburgh Airport to a friend or colleague?"].dropna().astype('int')
#%%
data["sentiments"] = [
    "positif" if element in [7, 8, 9, 10]
    else "neutre" if element in [4, 5, 6]
    else "negatif"
    for element in data["How likely are you to recommend Edinburgh Airport to a friend or colleague?"]
]
#%%
data_test=data.head(500)
#%%
print(data_test.head(), data_test.shape)

#%%
"""Creation of the interface for the text extraction part"""

df = pd.DataFrame(data_test)
#%%
# Interface utilisateur
st.title("Mini report of comments by individual profile")
st.subheader("If you do not want to select a certain filter, click 'None' ")

age = st.selectbox("Age",  np.append(df["What age group do you fall into?"].unique(), 'None'))
sexe = st.selectbox("Sexe", np.append(df["What is your gender?"].unique(), "None"))
origine = st.selectbox("Origine", np.append(df["What country do you live in?"].unique(), "None"))
religion= st.selectbox("Religion",np.append(df["What is your religion or belief?"].unique(), "None"))
Airline= st.selectbox("Airline", np.append(df["What Airline were you travelling with?"].unique(), "None"))


#%%
profil_neg = df[df["sentiments"] == "negatif"].copy()


# Apply filters only if the value is not "None"
if age != "None":
    profil_neg = profil_neg[profil_neg["What age group do you fall into?"] == age]
if sexe != "None":
    profil_neg = profil_neg[profil_neg["What is your gender?"] == sexe]
if origine != "None":
    profil_neg = profil_neg[profil_neg["What country do you live in?"] == origine]
if religion != "None":
    profil_neg = profil_neg[profil_neg["What is your religion or belief?"] == religion]
if Airline != "None":
    profil_neg = profil_neg[profil_neg["What Airline were you travelling with?"] == Airline]
#%%


profil_pos=df[df["sentiments"] == "positif"].copy()

# Apply filters only if the value is not "None"
if age != "None":
    profil_pos = profil_pos[profil_pos["What age group do you fall into?"] == age]
if sexe != "None":
    profil_pos = profil_pos[profil_pos["What is your gender?"] == sexe]
if origine != "None":
    profil_pos = profil_pos[profil_pos["What country do you live in?"] == origine]
if religion != "None":
    profil_pos = profil_pos[profil_pos["What is your religion or belief?"] == religion]
if Airline != "None":
    profil_pos = profil_pos[profil_pos["What Airline were you travelling with?"] == Airline]
#%%
profil_neutre=df[df["sentiments"] == "neutre"].copy()


# Apply filters only if the value is not "None"
if age != "None":
    profil_neutre = profil_neutre[profil_neutre["What age group do you fall into?"] == age]
if sexe != "None":
    profil_neutre = profil_neutre[profil_neutre["What is your gender?"] == sexe]
if origine != "None":
    profil_neutre = profil_neutre[profil_neutre["What country do you live in?"] == origine]
if religion != "None":
    profil_neutre = profil_neutre[profil_neutre["What is your religion or belief?"] == religion]
if Airline != "None":
    profil_neutre = profil_neutre[profil_neutre["What Airline were you travelling with?"] == Airline]

#%%

st.subheader("Summary of comments: Positive")

if not profil_pos.empty:
    texte = " ".join(profil_pos["Do you have any other feedback or suggestions?"].dropna().astype(str).tolist())
    if len(texte.split()) > 1024:
        texte = texte[:1025]
    if len(texte.split()) > 5:
        résumé = summarizer(texte, max_length=60, min_length=10,
do_sample=False)[0]["summary_text"]
        st.write(f"Customers matching this profile say that:*{résumé}*")
    else:
        st.write("Pas assez de contenu pour générer un résumé.")
else:
    st.write("Aucun commentaire pour ce profil sélectionné.")

#%%
st.subheader("Comment Summary: Neutral")

if not profil_neutre.empty:
    texte = " ".join(profil_neutre["Do you have any other feedback or suggestions?"].dropna().astype(str).tolist())
    if len(texte.split()) > 1024:
        texte = texte[:1025]
    if len(texte.split()) > 5:
        résumé = summarizer(texte, max_length=100, min_length=10,
do_sample=False)[0]["summary_text"]
        st.write(f"Customers matching this profile say that :*{résumé}*")
    else:
        st.write("Pas assez de contenu pour générer un résumé.")
else:
    st.write("Aucun commentaire pour ce profil sélectionné.")


#%%
st.subheader("Comment Summary: Negative")

if not profil_neg.empty:
    texte = " ".join(profil_neg["Do you have any other feedback or suggestions?"].dropna().astype(str).tolist())
    if len(texte.split()) > 1024:
        texte = texte[:1025]
    if len(texte.split()) > 5:
        résumé = summarizer(texte, max_length=100, min_length=10,
do_sample=False)[0]["summary_text"]
        st.write(f"Customers matching this profile say that :*{résumé}*")
    else:
        st.write("Not enough content to generate a summary.")
else:
    st.write("No comments for this selected profile.")

