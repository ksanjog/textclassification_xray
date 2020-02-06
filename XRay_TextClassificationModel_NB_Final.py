# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:05:59 2018

@author: Kumar Sanjog
"""

import pandas as pd
import string
import re
import numpy as np
import sys

#-----------------------------------------------------------------------
#Read input files

products=pd.read_csv(sys.argv[1], encoding='latin-1', engine= 'python')
#INPUT_PATH = "C:\\Users\\Kumar Sanjog\\Documents\\Projects\\X-Ray Classification Model\\Py\\"
#WORK_DIR = INPUT_PATH + "Python"
#products=pd.read_csv(INPUT_PATH+'products.csv',sep=";", encoding='latin-1', engine= 'python')


#-----------------------------------------------------------------------
new_punct = ''.join([i for i in string.punctuation if i not in ['-',' ']])

def func_remove_hyphen(x):
    x = re.sub("^-", "", x)# remove hyphen from start of sentence
    x = re.sub("-$", "", x)# remove hyphen from end of sentence
    x = re.sub("([a-zA-Z0-9])-([^a-zA-Z0-9])", "\\1\\2", x)
    x = re.sub("([^a-zA-Z0-9])-([a-zA-Z0-9])", "\\1\\2", x)
    x = re.sub("([a-zA-Z0-9])-([a-zA-Z0-9])", "\\1-\\2", x)
    x = re.sub("([^a-zA-Z0-9])-([^a-zA-Z0-9])", "\\1\\2", x)
    x = "".join(c for c in x if c not in new_punct)
    return(x)
    
products['PRODUCT'] = products['PRODUCT']

products['PRODUCT'] = products['PRODUCT'].apply(lambda x : func_remove_hyphen(x))

#Removing all multiple spaces in the string
products['PRODUCT'] = products['PRODUCT'].apply(lambda x : re.sub(" +", " ", x))

#Remove multiple spaces and dashes in the beginning and end
products['PRODUCT'] = products['PRODUCT'].str.lstrip(' -')
products['PRODUCT'] = products['PRODUCT'].str.rstrip(' -')

#Converting to lowercase
products['PRODUCT'] = products['PRODUCT'].str.lower()

#Remove blank products and keywords
products = products[products['PRODUCT'] != ""]

#Remove products with duplicate category
products=products.sort_values(['PRODUCT','MANUAL_CATEGORIZATION'],ascending=[True, True])
products=products.drop_duplicates(subset='PRODUCT',keep=False)

#Assign ids to manual categories
products['MANUAL_CATEGORIZATION']=products['MANUAL_CATEGORIZATION'].str.lower()
products['category_id'] = products['MANUAL_CATEGORIZATION'].rank(method='dense').astype(int)
products['category_id'] = products['category_id'] - 1


#Slice the data to keep only those categries that have above 100 data points (will be helpful to learn)
countrows = products.groupby(['category_id'])['PRODUCT'].count()
countrows = countrows[countrows > 100].index.tolist()
products = products[products['category_id'].isin(countrows)]

#Create new category_id column since a few categories would have gotten dropped from previous step
del products['category_id']
products['category_id'] = products['MANUAL_CATEGORIZATION'].rank(method='dense').astype(int)
products['category_id'] = products['category_id'] - 1

#Split the data in training and test sample - 80-20 split for all categories
from sklearn.model_selection import train_test_split

num_cat = max(products['category_id'])
train = pd.DataFrame()
test = pd.DataFrame()

for cur_cat in range(0,num_cat):
    curcatdata = products[(products.category_id==cur_cat) & (products.PRODUCT!='')]
    curtrain, curtest = train_test_split(curcatdata, test_size=0.2)
    
    train = train.append(curtrain)
    test = test.append(curtest)

del products, curtrain, curtest, countrows, num_cat, new_punct, cur_cat, curcatdata

#Naive Bayes Classification 
target = train.category_id.astype(int)
target = target.reset_index()
del target['index']
target = target.category_id.tolist()
targetnames = train.loc[:,['MANUAL_CATEGORIZATION','category_id']].drop_duplicates().MANUAL_CATEGORIZATION.tolist()
data = train.PRODUCT.tolist()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

#Using Multinomial algorithm for classification
from sklearn.naive_bayes import MultinomialNB
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
text_clf = text_clf.fit(data, target)

test['Predicted_Id_Multinomial'] = text_clf.predict(test.PRODUCT)
test['match_multin'] = np.where(test.Predicted_Id_Multinomial == test.category_id, 1, 0)
Precision_multin_test = sum(test.match_multin)/len(test.match_multin)

#Using SVM algorithm for classification
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
text_clf_svm = text_clf_svm.fit(data, target)

test['Predicted_Id_SVM'] = text_clf_svm.predict(test.PRODUCT)
test['match_svm'] = np.where(test.Predicted_Id_SVM == test.category_id, 1, 0)
Precision_svm_test = sum(test.match_svm)/len(test.match_svm)


#Using Grid search on Multinomial algorithm
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3),}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(data, target)

test['Predicted_Id_GSclf'] = gs_clf.predict(test.PRODUCT)
test['match_gsclf'] = np.where(test.Predicted_Id_GSclf == test.category_id, 1, 0)
Precision_gsmultin_test = sum(test.match_gsclf)/len(test.match_gsclf)

#Using Grid search on SVM algorithm
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3),}
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(data, target)

test['Predicted_Id_GSsvm'] = gs_clf_svm.predict(test.PRODUCT)
test['match_gssvm'] = np.where(test.Predicted_Id_GSsvm == test.category_id, 1, 0)
Precision_gssvm_test = sum(test.match_gssvm)/len(test.match_gssvm)