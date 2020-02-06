<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:31:25 2018

@author: Kumar Sanjog
"""

import pandas as pd
import string
import re
import numpy as np
import pickle
import sys


#-----------------------------------------------------------------------
#Read input files
products=pd.read_csv(sys.argv[1], sep=";", encoding='latin-1', engine= 'python')


#Min length criteria for new keyword identification
#MIN_NEW_KEYWORD_LENGTH = 2

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


categoryid = train.loc[:,['MANUAL_CATEGORIZATION','category_id']].drop_duplicates()


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

#Using Grid search on Multinomial algorithm
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3),}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(data, target)

test['Predicted_Id_GSclf'] = gs_clf.predict(test.PRODUCT)
test['match_gsclf'] = np.where(test.Predicted_Id_GSclf == test.category_id, 1, 0)
Precision_gsmultin_test = sum(test.match_gsclf)/len(test.match_gsclf)

#Output
categoryid.to_csv(sys.argv[2], sep=",",index=False)
Precision_gsmultin_test.to_csv(sys.argv[3], sep=",",index=False)

filename = 'finalized_xray_model.sav'
=======
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:31:25 2018

@author: Kumar Sanjog
"""

import pandas as pd
import string
import re
import numpy as np
import pickle
import sys


#-----------------------------------------------------------------------
#Read input files
products=pd.read_csv(sys.argv[1], sep=";", encoding='latin-1', engine= 'python')


#Min length criteria for new keyword identification
#MIN_NEW_KEYWORD_LENGTH = 2

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


categoryid = train.loc[:,['MANUAL_CATEGORIZATION','category_id']].drop_duplicates()


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

#Using Grid search on Multinomial algorithm
from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3),}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(data, target)

test['Predicted_Id_GSclf'] = gs_clf.predict(test.PRODUCT)
test['match_gsclf'] = np.where(test.Predicted_Id_GSclf == test.category_id, 1, 0)
Precision_gsmultin_test = sum(test.match_gsclf)/len(test.match_gsclf)

#Output
categoryid.to_csv(sys.argv[2], sep=",",index=False)
Precision_gsmultin_test.to_csv(sys.argv[3], sep=",",index=False)

filename = 'finalized_xray_model.sav'
>>>>>>> 1e3a403848e0d1bb178b793268a0d5d453f11e3f
pickle.dump(gs_clf, open(filename, 'wb'))