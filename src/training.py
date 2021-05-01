import sys
import pandas as pd
pd.options.display.max_columns = 30
import numpy as np
from time import time
#
import warnings 
warnings.filterwarnings('ignore')
#
import nltk
from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()
#from nltk.corpus import stopwords
#stop_words = set(stopwords.words('spanish'))
#
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer#, HashingVectorizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report#, ConfusionMatrixDisplay, plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils import parallel_backend
#
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#from xgboost import XGBClassifier
#
from joblib import dump, load
from datetime import datetime
now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

# get corpus
def get_dataset(file_,bal=False):
  if bal:
    file_ = file_ + "Bal"
  data = pd.read_table(file_+".txt", sep="\|\|\|", index_col=False,usecols=[0,1],names=['x','y'],dtype=str,engine='python',header=None) # or sep=tab
  data = data.sample(frac=1).reset_index(drop=True)
  data = data.fillna('0')
  data['x'] = data.x.str.strip()
  data['y'] = data.y.str.strip()
  return data
    
# obj to str
def clean_parens(text):
  return str(text)

def split_data(rutaRaiz, bal = False):

  file_ = rutaRaiz + "/ds/" + "sa3_train"
  trainDataset = get_dataset(file_,bal)
  print('Total amount of train','balanced',str(bal),len(trainDataset.index))

  file_ = rutaRaiz + "/ds/" + "sa3_dev"
  validationDataset = get_dataset(file_,bal)
  print('Total amount of dev','balanced',str(bal),len(validationDataset.index))

  file_ = rutaRaiz + "/ds/" + "sa3_test"
  testDataset = get_dataset(file_,bal)
  print('Total amount of test','balanced',str(bal),len(testDataset.index))

  return trainDataset, validationDataset, testDataset

def MyCustomTokenizer(x):
  tokenizer = TweetTokenizer() #RegexpTokenizer(r"(\w+\'\w?)|(\w+)")

  return tokenizer.tokenize(str(x)) #.lower() bal SVC

# prepare models pipeline
def benchmark(path, x, y, models, train_model=True, bal=False, model_target='all'):
    
    # 1.
    _train, _dev, _test = split_data(path,bal)
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test  =  _train[x], _train[y], _dev[x], _dev[y], _test[x], _test[y]
    
    # 2.
    pipeline = {}
    # iter
    for name, model in models.items():
        # specific model train/test
        if model_target not in [name,'all']:
            continue
            
        # Define a pipeline combining a text feature extractor with classifier
        pipeline[name] = Pipeline([
                ('vect', CountVectorizer(
                  analyzer = 'word',
                  tokenizer = MyCustomTokenizer,
                  lowercase = False,
                  ngram_range = (1,1), # 1,2 CNB ALL
                  #preprocessor=str,
                  min_df=3
                  )),
                ('tfidf', TfidfVectorizer(
                  analyzer = 'word',
                  tokenizer = MyCustomTokenizer,
                  lowercase = False,
                  ngram_range = (1,1), # 1,2 CNB ALL
                  #preprocessor=str,
                  min_df=3
                )),
                ('clf', model),
            ], verbose=1)
        
        print('... Processing', 'Balanced: ', bal)
        # train the model 
        with parallel_backend('threading'):
            if train_model:
                print('Init train {}'.format(name))
                pipeline[name].fit(X_train, Y_train)
                print('End train {}'.format(name))
        
        # save or load model
        if train_model:
            dump(pipeline[name], 'models/{}_bal{}_{}.joblib'.format(name,bal,now), compress=3 if name=='RFC' else 0) 
        else:
            pipeline[name] = load('models/{}_bal{}_{}.joblib'.format(name,bal,now)) 
        print('Save/load model {}_bal{}_{}'.format(name,bal,now))
        
        # test the model 
        with parallel_backend('threading'):
            # dev
            print("DEV")
            pred = pipeline[name].predict(X_dev)
            score1 = accuracy_score(Y_dev, pred)
            score2 = balanced_accuracy_score(Y_dev, pred)
            print("accuracy:   %0.3f" % score1)
            print("bal. accuracy:   %0.3f" % score2)
            #    
            print("classification report:")
            print(classification_report(Y_dev, pred))
            print("confusion matrix:")
            cm = confusion_matrix(Y_dev, pred)
            print(cm)
            #ConfusionMatrixDisplay(cm).plot()
            # test
            print("TEST")
            pred = pipeline[name].predict(X_test)
            score1 = accuracy_score(Y_test, pred)
            score2 = balanced_accuracy_score(Y_test, pred)
            print("accuracy:   %0.3f" % score1)
            print("bal. accuracy:   %0.3f" % score2)
            #    
            print("classification report:")
            print(classification_report(Y_test, pred))
            print("confusion matrix:")
            cm = confusion_matrix(Y_test, pred)
            print(cm)
            #ConfusionMatrixDisplay(cm).plot()
            
    return pipeline
    
# call from main   
def run_train(path, x, y, train_model=True, bal=False, model_target='all'):

    # define models
    models = {
          "CNB": ComplementNB(fit_prior=True, class_prior=None, alpha=0.1),
          "SVC": SVC(kernel='poly', class_weight='balanced'), # poly bal, sigmoid unbal
          "LogReg":LogisticRegression(solver='sag',n_jobs=-1),
          #"XGB":XGBClassifier(n_jobs=-1), # slow for large number of classes...
          "RFC":RandomForestClassifier(n_jobs=-1),
          "KNN":KNeighborsClassifier(n_neighbors=10,n_jobs=-1) # slow for large number of classes, use 10 neighbors
          } 

    # target
    benchmark(path, x, y, models, train_model, bal, model_target)
    
