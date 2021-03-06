from __future__ import print_function
import tensorflow as tf
from keras.callbacks import Callback
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
import psycopg2 as pg
from scipy.signal import butter, lfilter, freqz
import time
logging.basicConfig(filename='python_sql_learning.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
plt.switch_backend('agg')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)



dir_csv = '/home/ywang86/csv/'
window_back= 60
window_forward = 20
shift_test = 5
shift = 7
no_ccs = 1
dts = []
dgnscd_sequence_limit = 3
num_patient = 3000

setting_str = "window_back:"  + str(window_back) + " window_forward: " + str(window_forward) + " shift: "+ str(shift)
print(window_back,window_forward,shift)

logging.critical('%s, %s, %s', window_back,window_forward,shift)

def csv_df(csv_name):
	csv_dir = dir_csv + csv_name 
	df = pd.read_csv(csv_dir)
	return df




df_diag_thru_dt_list_all = csv_df('yw_diag_thru_dt_list_all.csv')
num_freq_icd = 150

conn = pg.connect("dbname='ldsbase'")
cursor = conn.cursor()
cursor.execute("SELECT claim_no,dsysrtky,sex,race FROM car_clm2007 limit 10")

print('start loading data')
df_hcpcs_thru_dngs_grouped_1115 = pd.read_sql('SELECT * FROM yw_hcpcs_thru_dngs_grouped_1115',conn)

df_hcpcs_icd__1115_freq = pd.read_sql('SELECT * FROM yw_hcpcs_icd__1115_freq_150', conn).head(num_freq_icd)

df_icd9_ccs_single_level_dx = pd.read_sql('SELECT * FROM icd9_ccs_single_level_dx', conn)

dict_icd_to_ccs = {x:y for [x, y] in df_icd9_ccs_single_level_dx[['icd9', 'ccs']].values.tolist()}
icd_freq_list = df_hcpcs_icd__1115_freq['line_icd_dgns_cd'].values.tolist()
list_icd_ccs =  icd_freq_list + list(set(df_icd9_ccs_single_level_dx['ccs'].values.tolist())) + ['naccs']  
df_codebook = pd.DataFrame({'code':list_icd_ccs})
df_codebook['index_col'] = df_codebook.index


dict_codebook = dict(zip(df_codebook.code, df_codebook.index_col))


print('finish loading data')
print(df_hcpcs_thru_dngs_grouped_1115.index.size)


df_hcpcs_thru_dngs_grouped_1115_sel = df_hcpcs_thru_dngs_grouped_1115.head(num_patient)

t1 = time.time()
df_grouped_icd = pd.read_sql("select * from yw_hcpcs_icd__1115_grouped_585x_all", conn)
t2 = time.time()
print('loading time for yw_hcpcs_icd__1115_grouped_585x_all is:', t2 - t1)




print("start reading data from csv")

logging.critical("start reading data from csv: %s", "new_df_training"+str(window_forward)+"_"+str(shift)+".csv")
logging.critical("start reading data from csv: %s", "new_df_testing"+str(window_forward)+"_"+str(shift)+".csv")


df_training = csv_df("new_df_training"+str(window_forward)+"_"+str(shift)+".csv")
df_testing = csv_df("new_df_testing"+str(window_forward)+"_"+str(shift_test)+".csv")


# df_training = csv_df("new_df_training.csv")
# df_testing = csv_df("new_df_testing.csv")


print("end reading data from csv")



# 
# print(df_all[list_icd_ccs])
# 
# clf.fit(df_all[list_icd_ccs],df_all['class']) 


# clf = LogisticRegression()

logging.critical('start training' )

clf = LogisticRegression()

# clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='lbfgs',beta_1=0.9, beta_2=0.999, early_stopping=False,epsilon=1e-08, hidden_layer_sizes=(7,4), learning_rate='constant',learning_rate_init=0.001, max_iter=10000, momentum=0.9,nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,warm_start=False)

y_training=df_training['class'].astype('int')
y_testing=df_testing['class'].astype('int')
y_training_interval = df_training['interval']
y_testing_interval = df_testing['interval']

print("y_training_interval is ",y_training_interval.values.tolist()[50:100])
print("y_testing_interval is ",y_testing_interval.values.tolist()[50:100])


y_list = y_testing.values.tolist()

clf.fit(df_training[list_icd_ccs],y_training)




plt.figure(3, figsize=(18, 18))


probs = clf.predict_proba(df_training[list_icd_ccs])
fpr, tpr, thresholds = roc_curve(y_training, probs[:,1])
auc_training = auc(fpr,tpr)
print('training auc is', auc_training)
logging.critical('training auc is %s', auc_training)
plt.plot(fpr,tpr, 'r', label='training')

probs = clf.predict_proba(df_testing[list_icd_ccs])
fpr, tpr, thresholds = roc_curve(y_testing, probs[:,1])
auc_testing = auc(fpr,tpr)
print('testing auc is', auc_testing)
logging.critical('testing auc is %s', auc_testing)
predicted_result = clf.predict(df_testing[list_icd_ccs])


plt.plot(fpr,tpr, 'b', label = 'testing')

plt.title("auc curve:  " + setting_str)
plt.savefig('./plots/auc_curve.png')


print("probs[:1] %s", probs[:,1])
print("predicted_result %s", predicted_result)

logging.critical("probs[:1] %s", probs[:,1])
logging.critical("predicted_result %s", predicted_result)

plt.figure(4, figsize = (72,18))

plt.subplot(411)
plt.plot(probs[:,1],'r*-',linewidth=4.0)
# plt.plot(predicted_result, 'b.-')
plt.plot(y_list, 'g>-')
plt.xlim(300,500)
plt.title("prob distributkon: " + setting_str)

plt.subplot(412)

plt.plot(probs[:,1],'r*-',linewidth=4.0)
# plt.plot(predicted_result, 'b.-')
plt.plot(y_list, 'g>-')
plt.xlim(500,700)
plt.title("prob distributkon: " + setting_str)

plt.subplot(413)

plt.plot(probs[:,1],'r*-',linewidth=4.0)
# plt.plot(predicted_result, 'b.-')
plt.plot(y_list, 'g>-')
plt.xlim(700,900)
plt.title("prob distributkon: " + setting_str)

plt.subplot(414)

plt.plot(probs[:,1],'r*-',linewidth=4.0)
# plt.plot(predicted_result, 'b.-')
plt.plot(y_list, 'g>-')
plt.title("prob distributkon: " + setting_str)

plt.savefig('./plots/result_prob.png')


	











	









