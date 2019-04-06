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

logging.basicConfig(filename='prediction.log_125', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


dir_csv = '/home/ywang86/csv/'
s3_s4_start_csv = 'yw_1115_min_s3_s4_positive_6m.csv'
grouped_diags = 'yw_1115_grouped_diags_withbilabel.csv'
# icd9_related_ckd = 'yw_icd9_list_ccs_49_50_156_157_158_161.csv'
icd9_related_ckd ='yw_icd_most_freq.csv'
icd9_to_ccs = 'icd9_ccs_single_level_dx.csv'
ccs_binary = 'ccs_id_binary.csv'
# record_all_csv =  'yw_1115_record_patients_s3_s4.csv'
pid_diags_dict = 'yw_pid_diags_dict_demo.csv'
icd9_related_ckd_5854 ='yw_icd_most_freq_5854.csv'

hcpcs_most_freq_5853 = 'yw_hcpcs_most_freq_5853.csv'
dict_pid_hcpcs_count = 'yw_dict_pid_hcpcs_count.csv'
patient_diag_thru = 'yw_dict_patient_diag_thru.csv'




thre_ave_pa = 0.125
s3_code = '5853'
window_size = 40
feature_num = 100
softmax_window = 60
max_predication = 479
binary_or_softmax = 0 # (binary: 0, softmax: 1)
balance_data = 0 # (0:orginial, 1: over sample to balance)
s1s2 = 0 # 0: no counting s1-s2, 1: counting s1-s2

def get_softmax_bits(max_predication,softmax_window):
	count = 0;
	value = int(max_predication/softmax_window)
	while (value > 0):
		count = count + 1
		value = value >> 1
	return count

softmax_bits = get_softmax_bits(max_predication,softmax_window)

def csv_df(csv_name):
	csv_dir = dir_csv + csv_name 
	df = pd.read_csv(csv_dir)
	return df
# select input features

# define data frame and dictionaries
df_s3_s4_start = csv_df(s3_s4_start_csv)
df_grouped_diags = csv_df(grouped_diags)
df_icd9_related_ckd = csv_df(icd9_related_ckd).head(100)
df_icd9_related_ckd_5854 = csv_df(icd9_related_ckd_5854).head(100)
df_hcpcs_most_freq_5853 = csv_df(hcpcs_most_freq_5853)
df_icd9_to_ccs = csv_df(icd9_to_ccs)
df_ccs_binary = csv_df(ccs_binary) 
# df_record_all = csv_df(record_all_csv) 
df_pid_diags_dict = csv_df(pid_diags_dict)
df_dict_pid_hcpcs_count = csv_df(dict_pid_hcpcs_count)
df_dict_patient_diag_thru = csv_df(patient_diag_thru)

dict_pid_diag = {(str(x),str(y)):z for [x, y, z] in df_pid_diags_dict[['dsysrtky', 'diag', 'min_thru_dt']].values.tolist()} 
dict_pid_hcpcs_count = {(str(x),str(y)):int(z) for [x, y, z] in df_dict_pid_hcpcs_count[['dsysrtky', 'hcpcs_cd', 'sum']].values.tolist()}
dict_patient_diag_thru = {(str(x)[0:9],str(y)):str(z) for [x, y, z] in df_dict_patient_diag_thru[['dsysrtky', 'dgnscd', 'min_thru_dt']].values.tolist()} 
dict_pid_s3_s4 =  {str(x): [y, z] for [x, y, z] in df_s3_s4_start[['dsysrtky', 'min_dt_s3', 'min_dt_s4']].values.tolist()} 

print(dict(list(dict_pid_s3_s4.items())[0:50]))

def days_between(d1, d2):
	d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
	d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
	return abs((d2 - d1).days)

	



def icd_convert(diag_list, dict_codebook,dict_icd9_ccs,dict_pid_diag,dict_pid_hcpcs_count,list_hcpcs_most_freq, pid, min_dt_s3, current_date):
	converted_diag = [0] * len(dict_codebook)
	for diag_old in diag_list:
		diag = diag_old.replace(" ", "")
		if diag == '[]':
			continue
		if diag in df_codebook.code.tolist():
			code = dict_codebook[diag]
		else:
			ccs_code = dict_icd9_ccs.get(diag) 
# 			ccs_code = df_icd9_to_ccs.loc(df_icd9_to_ccs['icd9'] == diag)
			if ccs_code is None:
				code = dict_codebook['naccs']
			else: 
				code = dict_codebook[ccs_code]
# 		converted_diag[code] = 1
		diag_days_to_s3 = days_between(current_date.strftime("%Y-%m-%d"), dict_pid_diag[(str(pid), str(diag))]) # only keep window_size record
					
		diag_days_to_s3 = min(diag_days_to_s3, window_size)
		if converted_diag[code] == 0:
			converted_diag[code] = diag_days_to_s3
		else:
			converted_diag[code] = min(diag_days_to_s3,converted_diag[code])
	# update hcpcs
	for hcpcs in list_hcpcs_most_freq:
		code = dict_codebook[hcpcs]
		if (str(pid),str(hcpcs)) in dict_pid_hcpcs_count.keys():
			converted_diag[code] = dict_pid_hcpcs_count[(str(pid),hcpcs)]

	if s1s2 == 0: 
		if ((str(pid), '5852') in dict_patient_diag_thru.keys()):
			if min_dt_s3 < dict_patient_diag_thru[(str(pid), '5852')]:
				converted_diag.append(days_between(min_dt_s3, '2007-01-01'))
			else:
				converted_diag.append(days_between(min_dt_s3, dict_patient_diag_thru[(str(pid), '5852')]))


		else:
			converted_diag.append(days_between(min_dt_s3, '2007-01-01'))

	else:	
		if ((str(pid), '5852') in dict_patient_diag_thru.keys()) and ((str(pid), '5851') in dict_patient_diag_thru.keys()) :
			converted_diag.append(days_between(min_dt_s3, dict_patient_diag_thru[(str(pid), '5852')]))
			converted_diag.append(days_between(dict_patient_diag_thru[(str(pid), '5852')], dict_patient_diag_thru[(str(pid), '5851')]))
		elif (str(pid), '5852') in dict_patient_diag_thru.keys():
			converted_diag.append(days_between(min_dt_s3, dict_patient_diag_thru[(str(pid), '5852')]))
# 			converted_diag.append(days_between(dict_patient_diag_thru[(str(pid), '5852')], '2007-01-01'))
			converted_diag.append(0)
		else:
			converted_diag.append(days_between(min_dt_s3, '2007-01-01'))
			converted_diag.append(days_between(min_dt_s3, '2007-01-01'))		
		
	return converted_diag




avg_duration = df_s3_s4_start['s3_s4_duration'].mean()

logging.critical("average duration from s3 to s4 is %s ",avg_duration)
threshold_durationg = thre_ave_pa * avg_duration



sliding_start_date = datetime.datetime.strptime("2011-6-30", "%Y-%m-%d")

sliding_end_date = datetime.datetime.strptime("2014-12-31", "%Y-%m-%d")


current_date = sliding_start_date

progression_month_low = -1

progression_month_high = 0



# ----------------------- prepare data -----------------------------------------------------------------------



# binary learning
def select_pat(originial_df_grouped_diags, binary_or_softmax, current_date, progression_month_low, progression_month_high):
	df_grouped_diags = originial_df_grouped_diags
	df_grouped_diags['befores4'] = df_grouped_diags.dsysrtky.apply(lambda x: (datetime.datetime.strptime(dict_pid_s3_s4[str(x)][1], "%Y-%m-%d")) > current_date)
	df_grouped_diags = df_grouped_diags.query('befores4 == True') 
	
# 	df_grouped_diags.dsysrtky.apply(lambda x:print(datetime.datetime.strptime(dict_pid_s3_s4[str(x)][0], "%Y-%m-%d") < current_date, datetime.datetime.strptime(dict_pid_s3_s4[str(x)][0], "%Y-%m-%d"), current_date))
	
	df_grouped_diags['afters3'] = df_grouped_diags.dsysrtky.apply(lambda x:( (datetime.datetime.strptime(dict_pid_s3_s4[str(x)][0], "%Y-%m-%d")+relativedelta.relativedelta(months=progression_month_low) <= current_date) and (datetime.datetime.strptime(dict_pid_s3_s4[str(x)][0], "%Y-%m-%d")+relativedelta.relativedelta(months=progression_month_high) >= current_date)))
	
	
	df_grouped_diags = df_grouped_diags.query('afters3 == True') 

	df_grouped_diags['to_s4_duration'] = df_grouped_diags.dsysrtky.apply(lambda x: days_between(dict_pid_s3_s4[str(x)][1],current_date.strftime("%Y-%m-%d")))
	
	if binary_or_softmax == 0:
		df_grouped_diags['bi_label'] = df_grouped_diags.to_s4_duration.apply(lambda x: x < threshold_durationg)
	else:
		df_grouped_diags['softmax_index'] = df_grouped_diags.to_s4_duration.apply(lambda x: int(x/softmax_window) if x < max_predication else int(max_predication/softmax_window))
		bit_format = '{0:0' + str(softmax_bits) + 'b}'
		df_grouped_diags['bi_label'] = df_grouped_diags.softmax_index.apply(lambda x: [int(d) for d in '{0:03b}'.format(x)])
	return  df_grouped_diags

def get_features(df_grouped_diags, current_date):
	df_grouped_diags.fillna('[]',inplace=True)
	df_grouped_diags['prncpal_group_list'] = df_grouped_diags.prncpal_group.apply(lambda x: x.split(','))
	df_grouped_diags['diags2_group_list'] = df_grouped_diags.diags2_group.apply(lambda x: x.split(','))
	df_grouped_diags['diags3_group_list'] = df_grouped_diags.diags3_group.apply(lambda x: x.split(','))
	df_grouped_diags['diags4_group_list'] = df_grouped_diags.diags4_group.apply(lambda x: x.split(','))
	df_grouped_diags['diags5_group_list'] = df_grouped_diags.diags5_group.apply(lambda x: x.split(','))
	df_grouped_diags['diags6_group_list'] = df_grouped_diags.diags6_group.apply(lambda x: x.split(','))
	df_grouped_diags['diags7_group_list'] = df_grouped_diags.diags7_group.apply(lambda x: x.split(','))
	df_grouped_diags['diags8_group_list'] = df_grouped_diags.diags8_group.apply(lambda x: x.split(','))
	df_grouped_diags['diags9_group_list'] = df_grouped_diags.diags9_group.apply(lambda x: x.split(','))
	df_grouped_diags['diags10_group_list'] = df_grouped_diags.diags10_group.apply(lambda x: x.split(','))
	df_grouped_diags['diags11_group_list'] = df_grouped_diags.diags11_group.apply(lambda x: x.split(','))
	df_grouped_diags['diags12_group_list'] = df_grouped_diags.diags12_group.apply(lambda x: x.split(','))
	df_grouped_diags['diags_all'] = df_grouped_diags.prncpal_group_list +df_grouped_diags.diags3_group_list+df_grouped_diags.diags4_group_list+df_grouped_diags.diags5_group_list+df_grouped_diags.diags6_group_list+df_grouped_diags.diags7_group_list+df_grouped_diags.diags8_group_list+df_grouped_diags.diags9_group_list+df_grouped_diags.diags10_group_list+df_grouped_diags.diags11_group_list+df_grouped_diags.diags12_group_list ;
	
	# Filter out diags after current date
	current_year = int(str(current_date).split('-')[0])
	current_month = int(str(current_date).split('-')[1])

	
	df_grouped_diags['yearno'] = df_grouped_diags.yearno.apply(lambda x: int(x))
	df_grouped_diags['monthno'] = df_grouped_diags.monthno.apply(lambda x: int(x))
		
	df_grouped_diags_s3_history = df_grouped_diags[(df_grouped_diags['yearno']<= current_year ) & (df_grouped_diags['monthno'] <= current_month)]
	
	df_all_diags_s3_history = df_grouped_diags_s3_history.groupby('dsysrtky',as_index=False).agg({'diags_all':'sum', 'min_dt_s3': 'max', 's3_s4_duration': 'max', 'bi_label': 'max'})

	df_all_diags_s3_history['diags_all_noduplicate'] = df_all_diags_s3_history.diags_all.apply(lambda x: list(filter(None,list(set(x)))))
	df_all_diags_s3_history['key_pid_diag'] = df_all_diags_s3_history[['dsysrtky', 'diags_all_noduplicate', 'min_dt_s3']].values.tolist()
	df_all_diags_s3_history['diag_duration'] =  df_all_diags_s3_history.key_pid_diag.apply(lambda x: icd_convert(x[1],dict_codebook,dict_icd9_ccs,dict_pid_diag,dict_pid_hcpcs_count,list_hcpcs_most_freq,int(x[0]), x[2], current_date))
	return df_all_diags_s3_history


# generate codebook
logging.critical('start codebook')

# logging.critical("ckd related icd9 codes are %s",df_icd9_related_ckd.icd9.tolist())


list_icd9_related_ckd_5853 = df_icd9_related_ckd.dgnscd.tolist()
list_icd9_related_ckd_5854 = df_icd9_related_ckd_5854.dgnscd.tolist()
list_icd9_related_ckd =list(set(list_icd9_related_ckd_5853 + list_icd9_related_ckd_5854))

list_hcpcs_most_freq = df_hcpcs_most_freq_5853.hcpcs_cd.tolist()


logging.critical("ckd related icd9 codes are %s",list_icd9_related_ckd)

list_icd9_related_ckd_max = [s+'_max' for s in list_icd9_related_ckd]

list_icd_ccs = list_icd9_related_ckd + df_ccs_binary.ccs.tolist() + ['naccs']  + list_hcpcs_most_freq # add a naccs for the icd that doesn't have and available ccs 

print("code book is %s", list_icd_ccs)
# list_icd_ccs = list_icd9_related_ckd + df_ccs_binary.ccs.tolist() + ['naccs'] + list_icd9_related_ckd_max # add a naccs for the icd that doesn't have and available ccs 
df_codebook = pd.DataFrame({'code':list_icd_ccs})
df_codebook['index_col'] = df_codebook.index



# dict_icd9_ccs = df_icd9_to_ccs.set_index('icd9').T.to_dict('list')

dict_icd9_ccs = dict(zip(df_icd9_to_ccs.icd9, df_icd9_to_ccs.ccs))

dict_codebook = dict(zip(df_codebook.code, df_codebook.index_col))
	

logging.critical('calculate diag_duration')


# # ----------------------- end of prepare data -----------------------------------------------------------------------
df_all = pd.DataFrame([])

while current_date < sliding_end_date:
	print("current_date is %s", current_date)
	df_grouped_diags_duplicate = select_pat(df_grouped_diags, binary_or_softmax, current_date, progression_month_low, progression_month_high)
	df_all_diags_s3_history = get_features(df_grouped_diags_duplicate, current_date)
	print(df_all_diags_s3_history.shape)
	df_all = df_all.append(df_all_diags_s3_history, ignore_index=True)
	print(df_all.shape)
	current_date = current_date + relativedelta.relativedelta(months=1)




# durationarray = np.array(df_all_diags_s3_history[['diag_duration']].tolist())
# print(durationarray.shape())
# print(durationarray[:,-1].sort())

save_csv = 1
if save_csv ==1 : 
	df_input = df_all['diag_duration']
	df_output = df_all[['bi_label', 's3_s4_duration']] 


	array_input = np.array(df_input.values.tolist()) 
	array_output = np.array(df_output.values.tolist())

	array_csv = np.concatenate((array_input, array_output), axis=1)

	pd.DataFrame(array_csv).to_csv("home/csv/df_all.csv", header=None)

	df_all = csv_df('df_all.csv')



logging.critical("start training")

tr, ts = train_test_split(df_all)

df_s1 = tr['diag_duration']
df_s2 = tr['bi_label']


if balance_data ==1: 
	print("resample data to balance negative and positive")
	df_s1_s2 = tr[['diag_duration','bi_label']]
	max_size = df_s1_s2['bi_label'].value_counts().max()
	# oversample imbalanced data:
	lst = [df_s1_s2]
	for class_index, group in df_s1_s2.groupby('bi_label'):
		lst.append(group.sample(int((max_size-len(group))), replace=True))
	df_new = pd.concat(lst)
		
	df_s1 = df_new['diag_duration']
	df_s2 = df_new['bi_label']
	


test_s1 = ts['diag_duration']
test_s2 = ts['bi_label']


array_s1 = np.array(df_s1.values.tolist()) 
array_s2 = np.array(df_s2.values.tolist())

# array_s1_s2 = np.concatenate((array_s1,array_s2, axis = 1)


# array_s1 = np.concatenate((array_s1,np.array(df_add.values.tolist())), axis = 1)

min_max_scaler = preprocessing.MinMaxScaler()
array_s1 = min_max_scaler.fit_transform(array_s1)



print( np.sum(array_s1, axis=0))


test_array_s1 = np.array(test_s1.values.tolist())
test_array_s2 = np.array(test_s2.values.tolist())
# test_array_s1 = np.concatenate((test_array_s1,np.array(test_add.values.tolist())), axis = 1)

test_array_s1 = preprocessing.scale(test_array_s1)
test_array_s1 = min_max_scaler.fit_transform(test_array_s1)

logging.critical(array_s1[0:10])

clf = Pipeline([
('feature_selection', SelectKBest(chi2, k=feature_num)),
('classification', MLPClassifier(activation='relu', alpha=1e-05, batch_size='lbfgs',beta_1=0.9, beta_2=0.999, early_stopping=False,epsilon=1e-08, hidden_layer_sizes=(7,4), learning_rate='constant',learning_rate_init=0.001, max_iter=10000, momentum=0.9,nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,warm_start=False))
])

clf.fit(array_s1, array_s2)   
                

logging.critical("finished training")


# ********************** RESULT DISPLAY ********************


if binary_or_softmax == 0: 
	print('training result')
	predicted_result = clf.predict(array_s1)
	predicted_prob = clf.predict_proba(array_s1)

	logging.critical("sum of predition is: %s",sum(predicted_result))
	logging.critical("sum of real result is: %s ",sum(array_s2))
	logging.critical("total number of sample is: %s ",len(array_s2))


	print("sum of predition is: %s",sum(predicted_result))
	print("sum of real result is: %s ",sum(array_s2))
	print("total number of sample is: %s ",len(array_s2))
	error_num_1 = 0
	error_num_0 = 0

	for i in range(len(array_s2)):
		if (predicted_result[i]!= array_s2[i]) and (array_s2[i] == 1):
			error_num_1 = error_num_1 + 1
		elif (predicted_result[i]!=array_s2[i]) and (array_s2[i] == 0):
			error_num_0 = error_num_0 + 1
			#logging.critical("predict",i, " wrong, real case is",array_s2[i]," with prob ", predicted_prob[i])
		#elif predicted_result[i] == 1:
		 #   logging.critical("predict", i, " right, real case is",array_s2[i]," with prob ", predicted_prob[i])
	accuracy_score = clf.score(array_s1, array_s2, sample_weight=None)
	logging.critical("accuracy_score is: %s ",accuracy_score)
	logging.critical("1-recall is %s ", error_num_1/len(array_s2))
	logging.critical("1-precision is %s ",error_num_0/len(array_s2))  
						   
	print("accuracy_score is: %s ",accuracy_score)
	print("1-recall is %s ", error_num_1/len(array_s2))
	print("1-precision is %s ",error_num_0/len(array_s2))  

	logging.critical('**********auc analysis***********')

	probs = clf.predict_proba(array_s1)
	fpr, tpr, thresholds = roc_curve(array_s2, probs[:,1])
	auc_rf = auc(fpr,tpr)

	logging.critical("print probs ")

	logging.critical(fpr) 
	logging.critical(tpr)
	logging.critical(thresholds)

	logging.critical('auc is %s', auc_rf)
	print('auc is %s', auc_rf)

	plt.switch_backend('agg')
	plt.figure(4, figsize=(18, 18))
	plt.plot(fpr,tpr, 'b')

	logging.critical('******************** TEST PART************************')
	print('******************** TEST PART************************')

	predicted_result = clf.predict(test_array_s1)
	predicted_prob = clf.predict_proba(test_array_s1)

	logging.critical("sum of predition is: %s ",sum(predicted_result))
	logging.critical("sum of real result is: %s ",sum(test_array_s2))
	logging.critical("total number of sample is: %s ",len(test_array_s2))
	
	print("sum of predition is: %s ",sum(predicted_result))
	print("sum of real result is: %s ",sum(test_array_s2))
	print("total number of sample is: %s ",len(test_array_s2))

	error_num_1 = 0
	error_num_0 = 0

	for i in range(len(test_array_s2)):
		if (predicted_result[i]!= test_array_s2[i]) and (test_array_s2[i] == 1):
			error_num_1 = error_num_1 + 1
		elif (predicted_result[i]!= test_array_s2[i]) and (test_array_s2[i] == 0):
			error_num_0 = error_num_0 + 1
			#logging.critical("predict",i, " wrong, real case is",array_s2[i]," with prob ", predicted_prob[i])
		#elif predicted_result[i] == 1:
		 #   logging.critical("predict", i, " right, real case is",array_s2[i]," with prob ", predicted_prob[i])
	accuracy_score = clf.score(test_array_s1, test_array_s2, sample_weight=None)
	logging.critical("accuracy_score is: %s ",accuracy_score)
	logging.critical("1-recall is" )
	logging.critical(error_num_1/len(test_array_s2))
	logging.critical("1-precision is %s ",error_num_0/len(test_array_s2))  


	print("accuracy_score is: %s ",accuracy_score)
	print("1-recall is" )
	print(error_num_1/len(test_array_s2))
	print("1-precision is %s ",error_num_0/len(test_array_s2))  
						   


	logging.critical('**********auc analysis***********')

	probs = clf.predict_proba(test_array_s1)
	fpr, tpr, thresholds = roc_curve(test_array_s2, probs[:,1])
	auc_rf = auc(fpr,tpr)
	logging.critical('auc is %s', auc_rf)
	print('auc is %s', auc_rf)

	plt.plot(fpr,tpr, 'r')
	plt.savefig('./plots/binaryroc.png')

else:
	predicted_result = clf.predict(array_s1)
	predicted_prob = clf.predict_proba(array_s1)

	logging.critical("sum of predition is: %s",sum(predicted_result))
	logging.critical("sum of real result is: %s ",sum(array_s2))
	logging.critical("total number of sample is: %s ",len(array_s2))


	error_num_1 = 0
	error_num_0 = 0

	# for i in range(len(array_s2)):
	#     if (predicted_result[i]!= array_s2[i]) and (array_s2[i] == 1):
	#         error_num_1 = error_num_1 + 1
	#     elif (predicted_result[i]!= array_s2[i]) and (array_s2[i] == 0):
	#     	error_num_0 = error_num_0 + 1
	#         #logging.critical("predict",i, " wrong, real case is",array_s2[i]," with prob ", predicted_prob[i])
	#     #elif predicted_result[i] == 1:
	#      #   logging.critical("predict", i, " right, real case is",array_s2[i]," with prob ", predicted_prob[i])
	accuracy_score = clf.score(array_s1, array_s2, sample_weight=None)

	logging.critical("accuracy_score is: %s ",accuracy_score)
	# logging.critical("1-recall is %s ", error_num_1/len(array_s2))
	# logging.critical("1-precision is %s ",error_num_0/len(array_s2))  
 
	print('******************** Training PART************************')
 
	print("accuracy_score is: %s ",accuracy_score)
	# print("1-recall is %s ", error_num_1/len(array_s2))
	# print("1-precision is %s ",error_num_0/len(array_s2))  
	# print("high use rate is %s ",sum(array_s2)/len(array_s2))                          


	logging.critical('**********auc analysis***********')

	probs = clf.predict_proba(array_s1)

	logging.critical('probs is %s', probs)
	a = [[int(''.join(str(x) for x in m), base=2) - int(''.join(str(x) for x in n), base=2),int(''.join(str(x) for x in m), base=2), int(''.join(str(x) for x in n), base=2) ] for m, n in list(zip(predicted_result,array_s2))]

	logging.critical('prediction compare to result  %s', list(zip(a, )))# fpr, tpr, thresholds = roc_curve(array_s2, probs[:,1])

	plt.switch_backend('agg')
	a = [[int(''.join(str(x) for x in m), base=2) - int(''.join(str(x) for x in n), base=2),int(''.join(str(x) for x in m), base=2), int(''.join(str(x) for x in n), base=2) ] for m, n in list(zip(predicted_result,array_s2))]
	b = np.array(a)

	count_correct = [0] * int(max_predication/softmax_window+1)
	count_rate = [0] * int(max_predication/softmax_window+1)
	count_all = [0]*int(max_predication/softmax_window+1)
	x_post = range(int(max_predication/softmax_window+1))
	for item in b:
		if abs(item[0]) < 2:
			count_correct[item[2]]=count_correct[item[2]]+1
		count_all[item[2]]=count_all[item[2]]+1

	for i in range(len(count_correct)):
		count_rate[i] = count_correct[i]/count_all[i]
	
	print("count_correct is %s ", count_correct)
	print("count_all is %s ",count_all)
	
	print("softmax to binary result ")
	for item in b:
		if (item[1]<7 and item[2]<7) or (item[1]==7 and item[2]==7):
			count_correct[item[2]]=count_correct[item[2]]+1
		count_all[item[2]]=count_all[item[2]]+1

	for i in range(len(count_correct)):
		count_rate[i] = count_correct[i]/count_all[i]
	
	logging.critical("count_correct is %s ", count_correct)
	logging.critical("count_all is %s ",count_all)
	plt.figure(2, figsize=(18, 18))
	plt.subplot(221)
	plt.hist(b[:,2], bins = int(max_predication/softmax_window+1))
	plt.title('Testing data s3-s4 duration distribution')

	plt.subplot(222)
	plt.hist(b[:,1], bins = int(max_predication/softmax_window+1))
	plt.title('Testing data s3-s4 prediction distribution')

	plt.subplot(223)
	plt.hist(b[:,0], bins = int(max_predication/softmax_window+1))
	plt.title('Testing data s3-s4 prediction error distribution')

	plt.subplot(224)
	plt.bar(x_post,count_rate)
	plt.title('Testing data s3-s4 Accuracy Rate')

	plt.bar(x_post,count_rate)
	plt.savefig('./plots/softmax_training.png')


	# auc_rf = auc(fpr,tpr)
	# logging.critical('auc is %s', auc_rf)
	# logging.critical('fpr and tpr are %s -- %s', fpr, tpr)

	print('******************** TEST PART************************')
	logging.critical('******************** TEST PART************************')
	predicted_result = clf.predict(test_array_s1)
	predicted_prob = clf.predict_proba(test_array_s1)

	# logging.critical("sum of predition is: %s ",sum(predicted_result))
	# logging.critical("sum of real result is: %s ",sum(test_array_s2))
	# logging.critical("total number of sample is: %s ",len(test_array_s2))

	error_num_1 = 0
	error_num_0 = 0

	# for i in range(len(test_array_s2)):
	#     if (predicted_result[i]!= test_array_s2[i]) and (test_array_s2[i] == 1):
	#         error_num_1 = error_num_1 + 1
	#     elif (predicted_result[i]!= test_array_s2[i]) and (test_array_s2[i] == 0):
	#     	error_num_0 = error_num_0 + 1
	#         #logging.critical("predict",i, " wrong, real case is",array_s2[i]," with prob ", predicted_prob[i])
	#     #elif predicted_result[i] == 1:
	#      #   logging.critical("predict", i, " right, real case is",array_s2[i]," with prob ", predicted_prob[i])
	accuracy_score = clf.score(test_array_s1, test_array_s2, sample_weight=None)
	logging.critical("accuracy_score is: %s ",accuracy_score)
	# logging.critical("1-recall is %s" ,error_num_1/len(test_array_s2))
	# logging.critical("1-precision is %s ",error_num_0/len(test_array_s2))  

 
 
	print("accuracy_score is: %s ",accuracy_score)
	# print("1-recall is %s" ,error_num_1/len(test_array_s2))
	# print("1-precision is %s ",error_num_0/len(test_array_s2))  
	# print("high use rate is %s ",sum(test_array_s2)/len(test_array_s2))                          


	logging.critical('**********auc analysis***********')

	probs = clf.predict_proba(test_array_s1)
	plt.switch_backend('agg')
	a = [[int(''.join(str(x) for x in m), base=2) - int(''.join(str(x) for x in n), base=2),int(''.join(str(x) for x in m), base=2), int(''.join(str(x) for x in n), base=2) ] for m, n in list(zip(predicted_result,test_array_s2))]
	b = np.array(a)

	count_correct = [0] * int(max_predication/softmax_window+1)
	count_rate = [0] * int(max_predication/softmax_window+1)
	count_all = [0]*int(max_predication/softmax_window+1)
	x_post = range(int(max_predication/softmax_window+1))
	for item in b:
		if abs(item[0]) < 2:
			count_correct[item[2]]=count_correct[item[2]]+1
		count_all[item[2]]=count_all[item[2]]+1

	for i in range(len(count_correct)):
		count_rate[i] = count_correct[i]/count_all[i]
	
	print("count_correct is %s ", count_correct)
	print("count_all is %s ",count_all)
	
	logging.critical("count_correct is %s ", count_correct)
	logging.critical("count_all is %s ",count_all)
	plt.figure(1, figsize=(18, 18))
	plt.subplot(221)
	plt.hist(b[:,2], bins = int(max_predication/softmax_window+1))
	plt.title('Testing data s3-s4 duration distribution')

	plt.subplot(222)
	plt.hist(b[:,1], bins = int(max_predication/softmax_window+1))
	plt.title('Testing data s3-s4 prediction distribution')

	plt.subplot(223)
	plt.hist(b[:,0], bins = int(max_predication/softmax_window+1))
	plt.title('Testing data s3-s4 prediction error distribution')

	plt.subplot(224)
	plt.bar(x_post,count_rate)
	plt.title('Testing data s3-s4 Accuracy Rate')


	plt.savefig('./plots/softmax_testing.png')
