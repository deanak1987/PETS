import pandas as pd

bank_df = pd.read_csv('dev_bank_dataset.csv')
swift_df = pd.read_csv('dev_swift_transaction_train_dataset.csv')
fraud_df = swift_df.loc[swift_df['Label'] == 1]
# fraud_df = pd.read_csv('mismatch.csv')


address_ids = set(bank_df['Street'])
fraud_df['AddressSenderExists'] = swift_df['OrderingStreet'].apply(lambda x : x in address_ids)
fraud_df['AddressReceiverExists'] = swift_df['BeneficiaryStreet'].apply(lambda x : x in address_ids)

def isFalse(sender, receiver):
    if sender is False or receiver is False:
        return 1
    else:
        return 0
fraud_df['wrongAddress'] = fraud_df.apply(lambda x: isFalse(x['AddressSenderExists'], x['AddressReceiverExists']), axis=1)
fraud_df.to_csv('mismatch.csv')
print(fraud_df)
print('Wrong Address = 1')
print(fraud_df['wrongAddress'].value_counts())
