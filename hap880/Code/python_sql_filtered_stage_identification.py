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
logging.basicConfig(filename='python_sql_filtered_stage_identification.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
plt.switch_backend('agg')

dir_csv = '/home/ywang86/csv/'
window_back= 60
window_forward = 20
shift_test = 3
shift = 3
no_ccs = 1
dts = []
dgnscd_sequence_limit = 3
num_patient = 20000

testing_delay_indicate = 'delay_'
training_delay_indicate = ''
balance = 1

print("change list")
print("move to more patients")
setting_str = "window_back:"  + str(window_back) + "|||| window_forward: " + str(window_forward) + "|||| shift: "+ str(shift)

print(window_back,window_forward,shift, num_patient)

logging.critical('%s, %s, %s', window_back,window_forward,shift)

def csv_df(csv_name):
	csv_dir = dir_csv + csv_name 
	df = pd.read_csv(csv_dir)
	return df


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def days_between(d1, d2):
	d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
	d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
	return abs((d2 - d1).days)

def get_stage_start_end(d_date, d_value):
	level_offset = 2
	num_days = days_between(d_date[-1], d_date[0]) 
	data = []
	t = range(num_days)
	for i in range(len(d_date)-1):
		level = int(d_value[i][3])-level_offset
		data.extend([level]*days_between(d_date[i+1], d_date[i]))


	# Filter requirements.
	order = 6
	fs = 60.0       # sample rate, Hz
	cutoff = 2.667  # desired cutoff frequency of the filter, Hz
	delay = int(fs/cutoff)-8
	t_filtered = range(0-delay, num_days-delay)
	m_th = 0.75

	# Get the filter coefficients so we can check its frequency response.
	b, a = butter_lowpass(cutoff, fs, order)

	# Plot the frequency response.
	w, h = freqz(b, a, worN=8000)

	# Filter the data, and plot both the original and filtered signals.
	y = butter_lowpass_filter(data, cutoff, fs, order)

	# find cross point:
	x = np.array(t_filtered)
	f_23 = np.array([2+m_th-level_offset]*num_days)
	f_34 = np.array([3+m_th-level_offset]*num_days)
	f_45 = np.array([4+m_th-level_offset]*num_days)
	f_56 = np.array([5+m_th-level_offset]*num_days)

	# f = np.arange(0, 1000)
	g = np.array(y)


	idx_23 = np.argwhere(np.diff(np.sign(f_23 - g))).flatten()-delay
	idx_34 = np.argwhere(np.diff(np.sign(f_34 - g))).flatten()-delay
	idx_45 = np.argwhere(np.diff(np.sign(f_45 - g))).flatten()-delay
	idx_56 = np.argwhere(np.diff(np.sign(f_56 - g))).flatten()-delay



	return x[idx_23],x[idx_34],x[idx_45],x[idx_56], num_days



def remove_nearby_swith(switch_list):
	remove_th =  60
	new_switch_list = list(switch_list)

	i = 0

	while i < len(new_switch_list)-1:
		if int(new_switch_list[i+1])-int(new_switch_list[i]) <= remove_th:
			current = new_switch_list[i]
			next = new_switch_list[i+1]
			new_switch_list.remove(current)
			new_switch_list.remove(next)
		else:
			i = i + 1
	return new_switch_list
	
	
def add_stage_switch_dt(pt):
	thru_dt_list = pt['thru_dt_list'].split(',')
	dgns_list = pt['dgns_list'].split(',')
	df_thru_dgns = pd.DataFrame({'thru_dt_list':thru_dt_list, 'dgns_list':dgns_list})
	df_thru_dgns = df_thru_dgns.sort_values(by=['thru_dt_list'])
	if df_thru_dgns['dgns_list'].values.tolist()[0] <='5853':
		idx_23, idx_34, idx_45,idx_56, num_days = get_stage_start_end(df_thru_dgns['thru_dt_list'].values.tolist(), df_thru_dgns['dgns_list'].values.tolist())
		return [idx_23, idx_34, idx_45, idx_56, [num_days]] 
	else:
		return [[-1],[-1],[-1],[-1],[-1]]

# def generate_data(df_selected): 

def convert_diag(diag_list, dict_icd_to_ccs, list_icd_ccs, dict_codebook):
	converted_list = [0]*len(list_icd_ccs)
	for diag in diag_list:
		if diag in list_icd_ccs:
			converted_list[dict_codebook[diag]] = 1
		elif (diag in dict_icd_to_ccs) and (dict_icd_to_ccs[diag] in list_icd_ccs):
			converted_list[dict_codebook[dict_icd_to_ccs[diag]]] = 1
		else:
			converted_list[dict_codebook['naccs']] = 1
	return converted_list
	
	

def generate_icd_list_within_window(p_icd, curr_time, back_time):
	thru_dt_list = p_icd['thru_dt_list'].split(',')
	dgns_list = p_icd['dgns_list'].split(',')
	df_thru_dgns = pd.DataFrame({'thru_dt':thru_dt_list, 'dgns':dgns_list})
	df_thru_dgns['thru_dt_dtformat'] = df_thru_dgns.apply(lambda x: datetime.datetime.strptime(x['thru_dt'], "%Y-%m-%d"), axis = 1)
	diag_list = []
	for i in range(df_thru_dgns.index.size):
		dt_dgns = df_thru_dgns.iloc[i]
		if (dt_dgns['thru_dt_dtformat'] <= curr_time and dt_dgns['thru_dt_dtformat']>= back_time):
			diag_list.append(dt_dgns['dgns'])
	diag_list = list(set(diag_list))
	converted_diag_list = convert_diag(diag_list, dict_icd_to_ccs, list_icd_ccs, dict_codebook)
	return converted_diag_list

def prepare_data(p_info, p_icd, window_back, window_forward, shift):
	pid = p_info['dsysrtky']
	in_cls = list_icd_ccs
	d_tmp = []
	for t in range(window_back, p_info['s34'][0], shift):
# 		print('s34 start time is ',p_info['s34'][0])
		curr_time = datetime.datetime.strptime(min(p_info['thru_dt_list'].split(',')), "%Y-%m-%d")+relativedelta.relativedelta(days=t)
		back_time = curr_time - relativedelta.relativedelta(days=window_back)
# 		print('current time is', curr_time)
# 		print('back time is', back_time)
		converted_diag_list = generate_icd_list_within_window(p_icd, curr_time, back_time)
		if p_info['s34'][0]-t >= window_forward:
			r_class = 0
		else:
			r_class = 1
		d_tmp.append(converted_diag_list+[r_class]+[p_info['s34'][0]-t]+[pid])
	df_columns = list_icd_ccs + ['class'] + ['interval'] + ['pid']
	df_p = pd.DataFrame(data = d_tmp, columns = df_columns)
	return df_p

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


start = time.time()
df_hcpcs_thru_dngs_grouped_1115_sel.loc[:,'stage_switch_dt'] = df_hcpcs_thru_dngs_grouped_1115_sel.apply(lambda x: add_stage_switch_dt(x),axis = 1)

df_hcpcs_thru_dngs_grouped_1115_sel.loc[:,'s23'] = df_hcpcs_thru_dngs_grouped_1115_sel.stage_switch_dt.apply(lambda x:remove_nearby_swith(x[0]))

df_hcpcs_thru_dngs_grouped_1115_sel.loc[:,'s23_positive'] = df_hcpcs_thru_dngs_grouped_1115_sel.s23.apply(lambda x: 1 if len(x) > 0 else 0)


df_hcpcs_thru_dngs_grouped_1115_sel.loc[:,'s34'] = df_hcpcs_thru_dngs_grouped_1115_sel.stage_switch_dt.apply(lambda x:remove_nearby_swith(x[1]))

df_hcpcs_thru_dngs_grouped_1115_sel.loc[:,'s34_positive'] = df_hcpcs_thru_dngs_grouped_1115_sel.s34.apply(lambda x: 1 if len(x) > 0 else 0)

df_hcpcs_thru_dngs_grouped_1115_sel.loc[:,'lod'] = df_hcpcs_thru_dngs_grouped_1115_sel.stage_switch_dt.apply(lambda x:x[4][-1])

df_hcpcs_thru_dngs_grouped_1115_sel.loc[:,'asce'] = df_hcpcs_thru_dngs_grouped_1115_sel.stage_switch_dt.apply(lambda x:0 if (len(x[0])>0 and x[0][0] == -1)  else 1)

df_hcpcs_thru_dngs_grouped_1115_asce = df_hcpcs_thru_dngs_grouped_1115_sel.loc[(df_hcpcs_thru_dngs_grouped_1115_sel['asce'] == 1)]

df_hcpcs_thru_dngs_grouped_1115_s3s4 = df_hcpcs_thru_dngs_grouped_1115_asce.loc[(df_hcpcs_thru_dngs_grouped_1115_sel['s34_positive'] == 1) & (df_hcpcs_thru_dngs_grouped_1115_sel['s23_positive'] == 1)]

df_hcpcs_thru_dngs_grouped_1115_s3s4.loc[:,'s3_duration'] = df_hcpcs_thru_dngs_grouped_1115_s3s4.stage_switch_dt.apply(lambda x:x[1][0] - x[0][0] )


# print('start generating s3s4')
# t1= time.time()
# df_hcpcs_thru_dngs_grouped_1115_s3s4.to_sql('yw_hcpcs_thru_dngs_grouped_1115_s3s4', conn)
# t2 = time.time()
# print('to sql time is: ', t2-t1)

# distrbution = df_hcpcs_thru_dngs_grouped_1115_s3s4['s3_duration'].values.tolist()
# 
# s3_list = df_hcpcs_thru_dngs_grouped_1115_s3s4['s23'].values.tolist()
# s4_list = df_hcpcs_thru_dngs_grouped_1115_s3s4['s34'].values.tolist()
# lod = df_hcpcs_thru_dngs_grouped_1115_s3s4['lod'].values.tolist()
# 
# 
# 		
# 
# s3_start = []
# s3_end = []
# 
# for item in s3_list:
# 	s3_start.append(item[0])
# 
# for item in s4_list:
# 	s3_end.append(item[0])
# 	
# plt.figure(1, figsize=(18, 54))
# 
# plt.subplot(411)
# plt.hist(distrbution, bins = 40, range=[59, 1500])
# plt.title('s3-s4 duration distribution')
# 
# plt.subplot(412)
# plt.scatter(s3_start, s3_end)
# plt.title('start end plot')
# 
# plt.subplot(413)
# plt.hist(lod)
# plt.title('length of duration for ckd')
# 
# plt.subplot(414)
# plt.hist(s3_end)
# plt.title('start end plot')
# 
# 
# plt.savefig('./plots/s3_duration.png')



print('number of patients with asce progression is %s', df_hcpcs_thru_dngs_grouped_1115_asce.index.size)

print('number of patients progressed from s3 to s4 is %s', df_hcpcs_thru_dngs_grouped_1115_s3s4.index.size)

end = time.time()
print('time to run stage identification: ', end - start)

tr, ts = train_test_split(df_hcpcs_thru_dngs_grouped_1115_s3s4)


# Index(['dsysrtky', 'thru_dt_list', 'dgns_list', 'stage_switch_dt', 's23',
#        's23_positive', 's34', 's34_positive', 'lod', 'asce', 's3_duration'],
#       dtype='object')



def combine_patient(df_patients, df_grouped_icd, window_back, window_forward, shift):
	frame = []
	for i in range(df_patients.index.size):
		print("the patients index is", i)
		logging.critical("the patients index is %s", i)
		p_info = df_patients.iloc[i]
		p_icd = df_grouped_icd.loc[df_grouped_icd['dsysrtky'] == p_info['dsysrtky']]
		df_p = prepare_data(p_info, p_icd.iloc[0], window_back, window_forward, shift)
		frame.append(df_p)
	df_all = pd.concat(frame,ignore_index=True)
	return df_all

try:
	df_training = combine_patient(tr, df_grouped_icd, window_back, window_forward, shift)
	df_testing = combine_patient(ts, df_grouped_icd, window_back, window_forward, shift_test)
except Exception as e:
    logging.Exception("Unexpected exception! %s",e)
    
    
print(df_training.shape)	
print(df_testing.shape)	



save_csv = 1
if save_csv ==1 : 
	training_file =  dir_csv+'blance'+ str(balance)+ testing_delay_indicate+ 'new_df_training'+str(window_forward)+'_'+str(shift)+'.csv'
	testing_file =  dir_csv+ 'blance'+ str(balance) + testing_delay_indicate+ 'new_df_training'+str(window_forward)+'_'+str(shift)+'.csv'
	df_training.to_csv(training_file)
	df_testing.to_csv(testing_file)
	logging.critical('training data is saved to file' + training_file)
	logging.critical('testing data is saved to file' +  testing_file )



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

y_list = y_testing.values.tolist()
print(y_list)
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
plt.title("auc curve" + setting_str)

plt.savefig('./plots/auc_curve.png')


print("probs[:1] %s", probs[:,1])
print("predicted_result %s", predicted_result)

logging.critical("probs[:1] %s", probs[:,1])
logging.critical("predicted_result %s", predicted_result)

plt.figure(4)
plt.plot(probs[:,1],'r*-')
plt.plot(predicted_result, 'b.-')
plt.plot(y_list, 'g>-')
plt.xlim(1,500)
plt.title("prob distributkon" + setting_str)

plt.savefig('./plots/result_prob.png')



	
	











	









