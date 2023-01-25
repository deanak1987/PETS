import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from diffprivlib.models import LogisticRegression as LogisticRegressionDP

print("Loading data")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Constructing features")

train['DiffCurrency'] = np.where(train['InstructedCurrency'] == train['SettlementCurrency'],0,1)
test['DiffCurrency'] = np.where(test['InstructedCurrency'] == test['SettlementCurrency'],0,1)

T = [-410285220,-410226720,-86880,0]

train['R1'] = np.where(train['InterimTime'] < T[0],1,0)
test['R1'] = np.where(test['InterimTime'] < T[0],1,0)

train['R2'] = np.where((train['InterimTime'] >= T[0]) & (train['InterimTime'] <= T[1]),1,0)
test['R2'] = np.where((test['InterimTime'] >= T[0]) & (test['InterimTime'] <= T[1]),1,0)

train['R3'] = np.where((train['InterimTime'] > T[1]) & (train['InterimTime'] < T[2]),1,0)
test['R3'] = np.where((test['InterimTime'] > T[1]) & (test['InterimTime'] < T[2]),1,0)

train['R4'] = np.where((train['InterimTime'] >= T[2]) & (train['InterimTime'] <= T[3]),1,0)
test['R4'] = np.where((test['InterimTime'] >= T[2]) & (test['InterimTime'] <= T[3]),1,0)

train['R5'] = np.where(train['InterimTime'] > T[3],1,0)
test['R5'] = np.where(test['InterimTime'] > T[3],1,0)


feature_cols= ['DiffCurrency','R1','R2','R3','R4','R5']

print(feature_cols)

X_train = train[feature_cols]
y_train = train['Label']
X_test = test[feature_cols]
y_test = test['Label']

print("Starting to train a LR model without DP")
clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred_train = clf.predict_proba(X_train)[:, 1]
print("LR on train data:", average_precision_score(y_train,y_pred_train))
y_pred = clf.predict_proba(X_test)[:, 1]
print("LR on test data:", average_precision_score(y_test,y_pred))

print("Starting to train a LR model with DP; epsilon = 5")
clf = LogisticRegressionDP(epsilon=5)
clf.fit(X_train,y_train)
y_pred_train = clf.predict_proba(X_train)[:, 1]
print("LR on train data:", average_precision_score(y_train,y_pred_train))
y_pred = clf.predict_proba(X_test)[:, 1]
print("LR on test data:", average_precision_score(y_test,y_pred))
