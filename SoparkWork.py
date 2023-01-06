import time
from pathlib import Path
import numpy as np
import pandas as pd
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
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext

from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext
import pandas as pd

# create the Spark Session
spark = SparkSession.builder.getOrCreate()

# create the Spark Context
sc = spark.sparkContext
train_bank_sc = spark.read.option("header", True).csv("dev_bank_dataset.csv")
train_bank_sc.show()
train_sc = spark.read.option("header",True).csv("dev_swift_transaction_train_dataset.csv")
train_sc.show()
test_sc = spark.read.option("header",True).csv("dev_swift_transaction_test_dataset.csv")
test_sc.show()
# Sets to make search faster
# These are NOT public values
bank_ids = set(train_bank_sc.Bank)

train_sc.withColumn('bankSenderExists', train_sc.Sender.lit(lambda x: x in train_bank_sc))