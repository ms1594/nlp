#!/bin/python3

import pandas as pd
import requests
from io import StringIO
import whois
import tldextract
import re
from datetime import datetime
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

############################# Malicious #######################################
url = 'https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links/output/domains/ACTIVE/list'
s = requests.get(url).content.decode()
phish = pd.read_csv(StringIO(s), skiprows=3, names=['url'])
del s, url
phish['url'] = phish['url'].apply(lambda x: x.split('//')[1].split('/')[0]) ### clean the url
phish['label'] = 1 

url = 'https://raw.githubusercontent.com/HexxiumCreations/threat-list/gh-pages/domainsonly'
s = requests.get(url).content.decode()
threat = pd.read_csv(StringIO(s), names=['url'])
del s, url
threat['label'] = 1

############################# Benign ##########################################
url = 'https://raw.githubusercontent.com/Kikobeats/top-sites/master/top-sites.json'
top_sites = pd.read_json(url)
del url
top_sites = top_sites[['rootDomain']].rename(columns={'rootDomain': 'url'})
top_sites['label'] = 0

### Combine dataframes
df = phish.append([threat, top_sites])
del  phish, threat, top_sites

### Initial Preprocessing
df = df.drop_duplicates(subset='url', keep='first').reset_index(drop=True)
df['url'] =  df['url'].apply(lambda x: x.strip('www.'))

### Extract suffix (tld)
df['suffix'] = df['url'].apply(lambda x: tldextract.extract(x).suffix)
### Domain name
df['domain'] =  list(map(lambda x, y: x.strip('.' + y), df['url'], df['suffix']))
### length of domain name
df['len_domain'] = df['domain'].apply(lambda x: len(x))
### Number of characters
df['num_char'] = df['domain'].apply(lambda x: len(''.join(re.split("[^a-zA-Z]*", x))))

### Expired or not
for i, u in enumerate(df['url']):
    print(i)
    try:
        df.loc[i, 'exp'] = (whois.whois(u)['expiration_date'][-1].date() < datetime.today().date()) + 0
    except:
        df.loc[i, 'exp'] = 0

### Categorical features Hashing (suffix, domain)
fh1 = FeatureHasher(n_features=32, input_type='string')
fh2 = FeatureHasher(n_features=256, input_type='string')
hash_suffix = fh1.fit_transform(df['suffix']).toarray()
hash_domain = fh2.fit_transform(df['domain']).toarray()
df = pd.concat([df, pd.DataFrame(hash_suffix), pd.DataFrame(hash_domain)], axis=1)
df = df.drop(['url', 'domain', 'suffix'], axis=1)
del hash_domain, hash_suffix, i, u, fh1, fh2

### write the created dataframe to pickle
df.to_pickle('df.pkl')

### load the dataframe from pickle
df = pd.read_pickle('df.pkl')

### Scale columns 
df['len_domain'] = (df['len_domain'] - df['len_domain'].min())/df['len_domain'].max()
df['num_char'] = (df['num_char'] - df['num_char'].min())/df['num_char'].max()

### split data into train-validation-test
X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'label'].values, 
                                                    df['label'].values, test_size=0.3, random_state=42)

################################# Randome Forest #########################################
for i in (50, 100, 200, 300):
    for j in (1, 2, 3, 5):
        for k in (6, 8, 10, 12, 14, 16, 18, 20):
            clf= RandomForestClassifier(n_estimators=i, min_samples_leaf=j, max_depth = k)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            print("trees= ", i , "min_leaf_sample= ", j , 'max_depth= ', k,'For RF classifier: TN: {} | FP: {} | FN: {} | TP: {}'.format(tn, fp, fn, tp))

################################# Logistic Regression #########################################
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred = clf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('For LR classifier: TN: {} | FP: {} | FN: {} | TP: {}'.format(tn, fp, fn, tp))


################################# KNN #########################################
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('For KNN classifier: TN: {} | FP: {} | FN: {} | TP: {}'.format(tn, fp, fn, tp))

################################# SVC #########################################
svc = SVC(C=1.0, kernel='rbf').fit(X_train, y_train)
svc_prediction = svc.predict(X_test) 
svc_cm = confusion_matrix(y_test, svc_prediction)
tn, fp, fn, tp = svc_cm.ravel()
print('For svc classifier: TN: {} | FP: {} | FN: {} | TP: {}'.format(tn, fp, fn, tp))


############################### MLP ###########################################
mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 10,),
                    activation='relu',
                    solver='adam',
                    learning_rate_init=0.001,
                    learning_rate = 'constant',
                    random_state=1, 
                    max_iter=100).fit(X_train, y_train)


mlp_prediction = mlp.predict(X_test) 
mlp_cm = confusion_matrix(y_test, mlp_prediction)
tn, fp, fn, tp = mlp_cm.ravel()
print('For mlp classifier: TN: {} | FP: {} | FN: {} | TP: {}'.format(tn, fp, fn, tp))
