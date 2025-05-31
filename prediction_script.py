import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler  
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report

import os

os.environ['TOKENIZERS_PARALLELISM'] = 'False'

TF_IDF = TfidfVectorizer()

def distribution(df,target,title=None):
    df_no_na = df.dropna(subset=[target]).copy()
    
    total = len(df_no_na)
    
    if target == 'Rates':
        plt.figure(figsize=(9, 5))
        n_color = df_no_na[target].nunique()
        ax = sns.countplot(x=target, data=df_no_na, hue = target,palette = sns.color_palette("hls", n_color), legend = False)
    else:
        plt.figure(figsize=(4, 4))
        ax = sns.countplot(x=target, data=df_no_na)

    plt.title(f"Distribution of {target} in {title}", fontsize=9)
    for p in ax.patches:
        bar_height = p.get_height()
        percentage = 100 * bar_height / total
        ax.annotate(f'{percentage:.1f}%', 
                    (p.get_x() + p.get_width() / 2., bar_height),
                    ha='center', va='bottom',
                    fontsize=9, color='black')

def metrics(xgb,x,y):
        
    y_pred = xgb.predict(x)
    y_prob = xgb.predict_proba(x)[:, 1]

    acc = accuracy_score(y,y_pred)
    auc = roc_auc_score(y,y_prob)

    fpr1, tpr1, thresholds = roc_curve(y, y_prob)
    
    return(fpr1, tpr1, acc, auc)

def plot_curve(xgb,df,target,column,sampler=None):
    
    X = TF_IDF.fit_transform(df[column])
    X = pd.DataFrame(X.toarray())
    
    Y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
  
    if sampler is not None:
        x_train, y_train = sampler.fit_resample(x_train, y_train)
        xgb.fit(x_train, y_train)
    else:
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        xgb.fit(x_train, y_train, sample_weight=sample_weights)

    fpr1, tpr1, acc_train, auc_train = metrics(xgb, x_train, y_train)
    fpr2, tpr2, acc_test, auc_test = metrics(xgb, x_test, y_test)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    if sampler is not None:
        fig.suptitle(f"Prediction of 'Satisfaction' by {sampler}", fontsize=10)
    else:
        fig.suptitle(f"Prediction of 'Satisfaction' by Sampling Weights", fontsize=10)

    # ROC curve for the train set
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].plot(fpr1, tpr1, label=f'AUC = {auc_train:.2f}')
    axes[0].set_title(f'ROC Curve - Train Set (Acc={acc_train:.2f})', fontsize = 8)
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].legend()

    # ROC curve for the test set
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].plot(fpr2, tpr2, label=f'AUC = {auc_test:.2f}', color='orange')
    axes[1].set_title(f'ROC Curve - Test Set (Acc={acc_test:.2f})', fontsize = 8)
    axes[1].set_xlabel('FPR')
    axes[1].set_ylabel('TPR')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def metrics_multiclass(xgb,x,y):
    y_pred = xgb.predict(x)
    y_prob = xgb.predict_proba(x)

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob, multi_class='ovr', average='macro')

    return (acc, auc)

def prediction(xgb, df, target, column, sampler=None):
    TF_IDF = TfidfVectorizer()

    Y = df[target]
    X = df[column]
    #X = TF_IDF.fit_transform(df[column])
    #X = pd.DataFrame(X.toarray())

    Y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42,stratify=Y)

    x_train = TF_IDF.fit_transform(x_train)
    x_test = TF_IDF.transform(x_test)

    svd = TruncatedSVD(n_components = 1000, random_state = 42)
    x_train = svd.fit_transform(x_train)
    x_test = svd.transform(x_test)

    if sampler is not None:
        x_train, y_train = sampler.fit_resample(x_train, y_train)
        xgb.fit(x_train, y_train)
    else:
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        xgb.fit(x_train, y_train, sample_weight=sample_weights)
    
    training_acc,training_auc = metrics_multiclass(xgb,x_train,y_train)
    test_acc,test_auc = metrics_multiclass(xgb,x_test,y_test)

    print(f"\nSampler : {sampler}\nTRAIN | Accuracy : {training_acc:.2f}, AUC : {training_auc:.2f}\nTEST  | Accuracy : {test_acc:.2f}, AUC : {test_auc:.2f}")
    


    