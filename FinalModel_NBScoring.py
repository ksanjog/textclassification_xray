<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:37:56 2018

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
categoryid = pd.read_csv(sys.argv[2], sep="," encoding='latin-1', engine= 'python')

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

test = products['PRODUCT'][1:2]
test = pd.DataFrame(test)

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
PredictedCategory = loaded_model.predict(test.PRODUCT)
PredictedCategory = pd.DataFrame(PredictedCategory)
PredictedCategory.columns = ['category_id']

result = pd.merge(PredictedCategory, categoryid, 'inner', on = 'category_id').MANUAL_CATEGORIZATION

=======
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 12:37:56 2018

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
categoryid = pd.read_csv(sys.argv[2], sep="," encoding='latin-1', engine= 'python')

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

test = products['PRODUCT'][1:2]
test = pd.DataFrame(test)

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
PredictedCategory = loaded_model.predict(test.PRODUCT)
PredictedCategory = pd.DataFrame(PredictedCategory)
PredictedCategory.columns = ['category_id']

result = pd.merge(PredictedCategory, categoryid, 'inner', on = 'category_id').MANUAL_CATEGORIZATION

>>>>>>> 1e3a403848e0d1bb178b793268a0d5d453f11e3f
result.tocsv(sys.argv[3],sep=",", index=False)