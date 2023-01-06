### Libraries for Data Handling
import time
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)

### Libraries for Algorithms

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.utils
from xgboost import XGBClassifier



tic = time.time()
# train = pd.read_csv("dev_swift_transaction_train_dataset.csv")
#

# train_bank = pd.read_csv("dev_bank_dataset.csv")
#
# # Sets to make search faster
# # These are NOT public values
# bank_ids = set(train_bank['Bank'])
#
# # Cannot check with single bank because it is possible that an end-end transaction has multiple individual transactions.
# # Such rows will have multiple banks ids but same Acctid and Name
# # Need to check individually
# acct_ids = set(train_bank['Account'])
#
# # Dictionary to make search faster
# acct_flag_search = train_bank[['Account','Flags']].set_index('Account').T.to_dict('list')
# account_name_search = train_bank[['Account','Name']].set_index('Account').T.to_dict('list')
#
# train["Timestamp"] = train["Timestamp"].astype("datetime64[ns]")
# test= pd.read_csv("dev_swift_transaction_test_dataset.csv")
# test["Timestamp"] = test["Timestamp"].astype("datetime64[ns]")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#
#
# train['bankSenderExists'] = train['Sender'].apply(lambda x : x in bank_ids)
# train['bankReceiverExists'] = train['Receiver'].apply(lambda x : x in bank_ids)
#
# train['isValidOrderingAcct'] = train['OrderingAccount'].apply(lambda x: x in acct_ids)
# train['isValidBeneficiaryAcct'] = train['BeneficiaryAccount'].apply(lambda x: x in acct_ids)


# test['bankSenderExists'] = test['Sender'].apply(lambda x : x in bank_ids)
# test['bankReceiverExists'] = test['Receiver'].apply(lambda x : x in bank_ids)
#
# test['isValidOrderingAcct'] = test['OrderingAccount'].apply(lambda x: x in acct_ids)
# test['isValidBeneficiaryAcct'] = test['BeneficiaryAccount'].apply(lambda x: x in acct_ids)

# Get Flags from bank
#
# train['OrderingFlag'] = train['OrderingAccount'].apply(lambda x: acct_flag_search[x][0]  if x in acct_flag_search.keys() else 0)
# train['BeneficiaryFlag'] = train['BeneficiaryAccount'].apply(lambda x: acct_flag_search[x][0] if x in acct_flag_search.keys() else 0)
#
# test['OrderingFlag'] = test['OrderingAccount'].apply(lambda x: acct_flag_search[x][0]  if x in acct_flag_search.keys() else 0)
# test['BeneficiaryFlag'] = test['BeneficiaryAccount'].apply(lambda x: acct_flag_search[x][0] if x in acct_flag_search.keys() else 0)
#
#
# # isAcctFlagged
# train['isOrderingAcctFlagged'] = train['OrderingFlag'].apply(lambda x: 0  if x==0 else 1)
# train['isBeneficiaryAcctFlagged'] = train['BeneficiaryFlag'].apply(lambda x: 0  if x==0 else 1)
# test['isOrderingAcctFlagged'] = test['OrderingFlag'].apply(lambda x:   0  if x==0 else 1)
# test['isBeneficiaryAcctFlagged'] = test['BeneficiaryFlag'].apply(lambda x: 0  if x==0 else 1)
#
# def check_name_oderingacct(x):
#     if x.OrderingAccount in account_name_search.keys():
#         if x.OrderingName == account_name_search[x.OrderingAccount][0]:
#             return True
#     return False
#
# def check_name_benefacct(x):
#     if x.BeneficiaryAccount in account_name_search.keys():
#         if x.BeneficiaryName == account_name_search[x.BeneficiaryAccount][0]:
#             return True
#     return False
#
#
#
# train['isOrderingNameCorrect'] = train[['OrderingAccount','OrderingName']].apply(lambda x: check_name_oderingacct(x), axis = 1)
# train['isBeneficiaryNameCorrect'] = train[['BeneficiaryAccount','BeneficiaryName']].apply(lambda x: check_name_benefacct(x), axis = 1)
#
# test['isOrderingNameCorrect'] = test[['OrderingAccount','OrderingName']].apply(lambda x: check_name_oderingacct(x), axis = 1)
# test['isBeneficiaryNameCorrect'] = test[['BeneficiaryAccount','BeneficiaryName']].apply(lambda x: check_name_benefacct(x), axis = 1)

# def check_name_order(x):
#     if x.isValidOrderingAcct == True and pd.isna(x.OrderingName):
#         return True
#     else:
#         return x.isOrderingNameCorrect
#
# def check_name_benef(x):
#     if x.isValidBeneficiaryAcct == True and pd.isna(x.BeneficiaryName):
#         return True
#     else:
#         return x.isBeneficiaryNameCorrect
#
#
# train['isOrderingNameCorrect'] = train[['isValidOrderingAcct','OrderingName','isOrderingNameCorrect']].apply(lambda x: check_name_order(x), axis = 1)
# train['isBeneficiaryNameCorrect'] = train[['isValidBeneficiaryAcct','BeneficiaryName','isBeneficiaryNameCorrect']].apply(lambda x: check_name_benef(x), axis = 1)
#
# test['isOrderingNameCorrect'] = test[['isValidOrderingAcct','OrderingName','isOrderingNameCorrect']].apply(lambda x: check_name_order(x), axis = 1)
# test['isBeneficiaryNameCorrect'] = test[['isValidBeneficiaryAcct','BeneficiaryName','isBeneficiaryNameCorrect']].apply(lambda x: check_name_benef(x), axis = 1)

#
# # Hour
# train["Timestamp"] = pd.to_datetime(train.Timestamp, errors='coerce')
# test["Timestamp"] = pd.to_datetime(test.Timestamp, errors='coerce')
# train["hour"] = train["Timestamp"].dt.hour
# test["hour"] = test["Timestamp"].dt.hour
#
# # Hour frequency for each sender
# senders = train["Sender"].unique()
# train["sender_hour"] = train["Sender"] + train["hour"].astype(str)
# test["sender_hour"] = test["Sender"] + test["hour"].astype(str)
# sender_hour_frequency = {}
# for s in senders:
#     sender_rows = train[train["Sender"] == s]
#     for h in range(24):
#         sender_hour_frequency[s + str(h)] = len(sender_rows[sender_rows["hour"] == h])
#
# train["sender_hour_freq"] = train["sender_hour"].map(sender_hour_frequency)
# test["sender_hour_freq"] = test["sender_hour"].map(sender_hour_frequency)

# Sender-Currency Frequency and Average Amount per Sender-Currency
# train["sender_currency"] = train["Sender"] + train["InstructedCurrency"]
# test["sender_currency"] = test["Sender"] + test["InstructedCurrency"]
#
# sender_currency_freq = {}
# sender_currency_avg = {}
#
# sc_set = set(
#     list(train["sender_currency"].unique()) + list(test["sender_currency"].unique())
# )
#
#
# for sc in sc_set:
#     sender_currency_avg[sc] = train[train["sender_currency"] == sc][
#         "InstructedAmount"
#     ].mean()
#
#
# for sc in sc_set:
#     sender_currency_freq[sc] = len(train[train["sender_currency"] == sc])
#
#
# train["sender_currency_freq"] = train["sender_currency"].map(sender_currency_freq)
# test["sender_currency_freq"] = test["sender_currency"].map(sender_currency_freq)
#
# train["sender_currency_amount_average"] = train["sender_currency"].map(
#     sender_currency_avg
# )
# test["sender_currency_amount_average"] = test["sender_currency"].map(sender_currency_avg)
#
#
# # Sender-Receiver Frequency
# train["sender_receiver"] = train["Sender"] + train["Receiver"]
# test["sender_receiver"] = test["Sender"] + test["Receiver"]
#
# sender_receiver_freq = {}
#
# for sr in set(
#         list(train["sender_receiver"].unique()) + list(test["sender_receiver"].unique())
# ):
#     sender_receiver_freq[sr] = len(train[train["sender_receiver"] == sr])
#
# train["sender_receiver_freq"] = train["sender_receiver"].map(sender_receiver_freq)
# test["sender_receiver_freq"] = test["sender_receiver"].map(sender_receiver_freq)
#
#
# #beneficiary_account_num_transactions =  train.groupby("BeneficiaryAccount")["UETR"].unique().apply(lambda l: len(l))
# #train["beneficiary_account_num_transactions"] = train["BeneficiaryAccount"].map(beneficiary_account_num_transactions)
# #test["beneficiary_account_num_transactions"] = test["BeneficiaryAccount"].map(beneficiary_account_num_transactions)
#
# SWIFT_COLS = ['SettlementAmount', 'InstructedAmount', 'hour',
#               'sender_hour_freq', 'sender_currency_freq',
#               'sender_currency_amount_average', 'sender_receiver_freq']
#
# scaler = StandardScaler()
#
# train[SWIFT_COLS] = scaler.fit_transform(train[SWIFT_COLS])
# test[SWIFT_COLS] = scaler.transform(test[SWIFT_COLS])

Y_train = train["Label"].values
Y_test = test["Label"].values

X_train_SWIFT = train[['SettlementAmount', 'InstructedAmount',  'hour',
                       'sender_hour_freq', 'sender_currency_freq',
                       'sender_currency_amount_average', 'sender_receiver_freq','isBeneficiaryAcctFlagged']].values


X_test_SWIFT = test[['SettlementAmount', 'InstructedAmount',  'hour',
                     'sender_hour_freq', 'sender_currency_freq',
                     'sender_currency_amount_average', 'sender_receiver_freq','isBeneficiaryAcctFlagged']].values
#
# xgb_SWIFT = XGBClassifier(n_estimators=100, maxrandom_state=0)
from sklearn.pipeline import make_pipeline
xgb_SWIFT = make_pipeline(StandardScaler(), XGBClassifier(learning_rate=0.1, max_depth=9,
                                                          n_estimators=100))

xgb_SWIFT.fit(X_train_SWIFT, Y_train)
pred_xgb = xgb_SWIFT.predict(X_test_SWIFT)
print("XGB Classification Report=\n\n", classification_report(Y_test, pred_xgb))
print("XGB Confusion Matrix=\n\n", confusion_matrix(Y_test, pred_xgb))
pred_proba_xgb = xgb_SWIFT.predict_proba(X_test_SWIFT)[:, 1]

print("AUPRC:", metrics.average_precision_score(y_true=Y_test, y_score=pred_proba_xgb))

prob_features = ['invalid_details', 'pred_proba_swift']

# code can be made in one line
# splittinf for clarity
def get_invalid_from_banks(x):
    #invalid = 0
    # Match details from bank and SWIFT
    if x.isValidOrderingAcct==False or x.isValidBeneficiaryAcct==False or x.isOrderingNameCorrect==False or x.isBeneficiaryNameCorrect==False:
        return 1
    # Fetch details from bank
    if x.isOrderingAcctFlagged == 1 or x.isBeneficiaryAcctFlagged == 1:
        return 1
    return 0


train['invalid_details'] = train[['isValidOrderingAcct',
                                  'isValidBeneficiaryAcct', 'isOrderingNameCorrect',
                                  'isBeneficiaryNameCorrect','isOrderingAcctFlagged','isBeneficiaryAcctFlagged']].apply(lambda x : get_invalid_from_banks(x), axis = 1)


test['invalid_details'] = test[['isValidOrderingAcct',
                                'isValidBeneficiaryAcct', 'isOrderingNameCorrect',
                                'isBeneficiaryNameCorrect','isOrderingAcctFlagged','isBeneficiaryAcctFlagged']].apply(lambda x : get_invalid_from_banks(x), axis = 1)
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)


ensemble_df = pd.DataFrame()
ensemble_df['SWIFT'] = pred_proba_xgb
ensemble_df['BANK'] = test['invalid_details'].values
ensemble_df['SWIFT+BANK'] = ensemble_df[['BANK','SWIFT']].max(axis=1)

# ensemble_df = pd.read_csv('ensemble.csv')
def to_bin(x):
    if x < 0.5:
        return 0
    else:
        return 1

ensemble_df['SBbin'] = ensemble_df['SWIFT+BANK'].apply(lambda x: to_bin(x))
ensemble_df.to_csv('ensemble.csv', index=False)
print("XGB Confusion Matrix=\n\n", confusion_matrix(Y_test, ensemble_df.SBbin))
print("AUPRC:", metrics.average_precision_score(y_true=Y_test, y_score=ensemble_df['SWIFT+BANK'].values))

toc = time.time()

print(toc - tic)