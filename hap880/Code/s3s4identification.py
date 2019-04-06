from __future__ import print_function
import tensorflow as tf
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
import sys
import io
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import json
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
import datetime
import logging
from dateutil import relativedelta
from dateutil import relativedelta
import sqlite3




logging.basicConfig(filename='prediction_feature_selection_25_44.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

plt.switch_backend('agg')

exp_duration = 90
resolution_duration = 3
diag_threshold = 0.9
today_date = "2019-03-19"
s3_code = '5853'
s4_code = '5854'

# parameters definition
def decay_by_time(duration):
	if duration > exp_duration:
		decay = 0
	else:
		decay = 1
	return decay

def credit_by_iter(current_credit, duration):
	if duration <= resolution_duration:
		new_credit = current_credit*decay_by_time(duration)
	else:
		new_credit = (current_credit + 1)*decay_by_time(duration)
	return new_credit
	
	

def csv_df(csv_name):
	csv_dir = dir_csv + csv_name 
	df = pd.read_csv(csv_dir)
	return df
	
def days_between(d1, d2):
	d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
	d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
	return abs((d2 - d1).days)
	
def diag_identification_date(dt_list):
	i = 0
	start_date = today_date
	for date in dt_list:

		date = date.replace(" ", "")
		if i == 0:
			current_credit = 0
		else: 
			duration = days_between(date, current_date) 
			current_credit = credit_by_iter(current_credit, duration)
		current_date = date
		if np.tanh(current_credit) > diag_threshold:
			start_date = current_date
			return start_date
		i = i + 1
	return start_date
			

# conn=sqlite3.connect('ldsbase.db') # enter full path here
# cursor = conn.cursor()
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# print(cursor.fetchall())
# df_dict_patient_diag_thru_all = pd.read_sql('select * from yw_dict_patient_diag_thru_all', conn)
	

dir_csv = '/home/ywang86/csv/'
df_diag_thru_dt_list_all = csv_df('yw_diag_thru_dt_list_all.csv')
df_s3_s4_start = csv_df('yw_1115_min_s3_s4_positive_6m.csv')

# df_patient_diag_thru columns: dsysrtky  | dgnscd |thru_dt_list

dict_diag_thru_dt_list_all = {(str(x),str(y)):sorted(z.split(',')) for [x, y, z] in df_diag_thru_dt_list_all[['dsysrtky', 'dgnscd', 'thru_dt_list']].values.tolist()}

# print(dict(list(dict_diag_thru_dt_list_all.items())[0:10]))


df_s3_s4_start['start'+s3_code] = df_s3_s4_start.dsysrtky.apply(lambda x:diag_identification_date(dict_diag_thru_dt_list_all[str(x), s3_code]))

df_s3_s4_start['start'+s4_code] = df_s3_s4_start.dsysrtky.apply(lambda x:diag_identification_date(dict_diag_thru_dt_list_all[str(x), s4_code]))

df_s3_s4_start['new_s3s4_duration'] = df_s3_s4_start.dsysrtky.apply(lambda x:days_between(diag_identification_date(dict_diag_thru_dt_list_all[str(x), s4_code]), diag_identification_date(dict_diag_thru_dt_list_all[str(x), s3_code])))

df_s3_s4_start['valid_s3_s4'] =  df_s3_s4_start.dsysrtky.apply(lambda x: 1 if (diag_identification_date(dict_diag_thru_dt_list_all[str(x), s4_code])!= today_date and diag_identification_date(dict_diag_thru_dt_list_all[str(x), s3_code])!= today_date) else 0)

old_s3 = lambda x: days_between(x['start5854'],x['min_dt_s3'])
df_s3_s4_start['old_s3_duration'] = df_s3_s4_start.apply(old_s3,axis=1)

# df_s3_s4_start['s3s4_duration'] = df_s3_s4_start.apply(lambda x: x['start'+s3_code] - x['start'+s4_code])



df_s3_s4 = df_s3_s4_start[df_s3_s4_start.valid_s3_s4 == 1]

avg_duration = df_s3_s4['old_s3_duration'].mean()

print("average duration from s3 to s4 is %s ",avg_duration)

print(df_s3_s4.shape)

df_s3_s4_30 = df_s3_s4[df_s3_s4_start.new_s3s4_duration <= 30 ]
print(df_s3_s4_30)

print(df_s3_s4[['min_dt_s3',   'min_dt_s4','s3_s4_duration', 'start5853',   'start5854',  'new_s3s4_duration', 'valid_s3_s4', 'old_s3_duration']].head(40))

distrbution = df_s3_s4['old_s3_duration'].values.tolist()
plt.figure(1, figsize=(18, 36))

plt.subplot(211)

plt.hist(distrbution, bins = 40, range=[0, 200])
plt.title('total data s3-s4 weighted duration distribution within 200 days')

plt.subplot(212)
plt.hist(distrbution, bins = 40)
plt.title('total data s3-s4 weighted duration distribution')
plt.savefig('./plots/weighted_distribution.png')





