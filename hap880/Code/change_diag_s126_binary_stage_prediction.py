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


logging.basicConfig(filename='prediction_feature_selection_25_44.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


dir_csv = '/home/ywang86/csv/'
# icd9_related_ckd = 'yw_icd9_list_ccs_49_50_156_157_158_161.csv'
ccs_binary = 'ccs_id_binary.csv'
# record_all_csv =  'yw_1115_record_patients_s3_s4.csv'
pid_diags_dict = 'yw_pid_diags_dict_demo.csv'
icd9_related_ckd_5854 ='yw_icd_most_freq_5854.csv'

hcpcs_most_freq_5853 = 'yw_hcpcs_most_freq_5853.csv'
dict_pid_hcpcs_count = 'yw_dict_pid_hcpcs_count.csv'
patient_diag_thru = 'yw_dict_patient_diag_thru.csv'



thre_ave_pa = 0.25
print("current threshold is %s", thre_ave_pa)
s3_code = '5853'
window_size = 40
feature_num = 100
softmax_window = 60
max_predication = 479
binary_or_softmax = 1 # (binary: 0, softmax: 1)
balance_data = 0 # (0:orginial, 1: over sample to balance)
s1s2 = 1 # 0: no counting s1-s2, 1: counting s1-s2
old_hcpcs = 1
no_ccs = 0
old_diag = 1
number_diags = 100
beyond_s3 = 1
cut_off_ahead = 30
print("s1s2 is s%", s1s2)
print("old_hcpcs is s%", old_hcpcs)
print("no_ccs is s%", no_ccs)
print("old_diag is s%", old_diag)
print("binary_or_softmax is s%", binary_or_softmax)
print("number_diags is s%", number_diags)
print("beyond_s3 is s%", beyond_s3)



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

df_s3_s4_start = csv_df('yw_1115_min_s3_s4_positive_6m.csv')
df_grouped_diags = csv_df('yw_1115_grouped_diags_withbilabel.csv')
df_icd9_related_ckd = csv_df('yw_icd_most_freq.csv').head(number_diags)
df_icd9_related_ckd_5854 = csv_df(icd9_related_ckd_5854).head(number_diags)

df_hcpcs_most_freq_5853 = csv_df(hcpcs_most_freq_5853)

df_icd9_to_ccs = csv_df('icd9_ccs_single_level_dx.csv')
df_ccs_binary = csv_df(ccs_binary) 
# df_record_all = csv_df(record_all_csv) 
df_pid_diags_dict = csv_df(pid_diags_dict)

df_dict_pid_hcpcs_count = csv_df(dict_pid_hcpcs_count)

df_hcpcs_thru_dt_list = csv_df('yw_hcpcs_thru_dt_list_all.csv')
df_diag_thru_dt_list = csv_df('yw_diag_thru_dt_list_all.csv')



logging.critical('total number of patients is ')
logging.critical(df_s3_s4_start.shape[0])

df_dict_patient_diag_thru = csv_df(patient_diag_thru)


cls = df_s3_s4_start.columns.tolist()

cls = df_grouped_diags.columns.tolist()

logging.critical(df_pid_diags_dict.head(10))



dict_pid_diag = {(str(x),str(y)):z for [x, y, z] in df_pid_diags_dict[['dsysrtky', 'diag', 'min_thru_dt']].values.tolist()} 

dict_pid_hcpcs_count = {(str(x),str(y)):int(z) for [x, y, z] in df_dict_pid_hcpcs_count[['dsysrtky', 'hcpcs_cd', 'sum']].values.tolist()}


dict_patient_diag_thru = {(str(x)[0:9],str(y)):str(z) for [x, y, z] in df_dict_patient_diag_thru[['dsysrtky', 'dgnscd', 'min_thru_dt']].values.tolist()} 


dict_hcpcs_thru_dt_list = {(str(x),str(y)):z.split(',') for [x, y, z] in df_hcpcs_thru_dt_list[['dsysrtky', 'hcpcs_cd', 'thru_dt_list']].values.tolist()}

dict_diag_thru_dt_list = {(str(x),str(y)):z.split(',') for [x, y, z] in df_diag_thru_dt_list[['dsysrtky', 'dgnscd', 'thru_dt_list']].values.tolist()}


# ----------------------- prepare data -----------------------------------------------------------------------
df_s3_s4_start = df_s3_s4_start[df_s3_s4_start.s3_s4_duration >= cut_off_ahead]

avg_duration = df_s3_s4_start['s3_s4_duration'].mean()

print("average duration from s3 to s4 is %s ",avg_duration)
logging.critical("average duration from s3 to s4 is %s ",avg_duration)

# binary learning

# ave_duration = 359.99681511174566

threshold_durationg = thre_ave_pa * avg_duration

print("shape of df_grouped_diags before removing", cut_off_ahead, " days is", df_grouped_diags.shape)
df_grouped_diags = df_grouped_diags[df_grouped_diags.s3_s4_duration >= cut_off_ahead]
print("shape of df_grouped_diags after removing", cut_off_ahead, " days is", df_grouped_diags.shape)


df_grouped_diags = df_grouped_diags.assign(bi_label=(df_grouped_diags.s3_s4_duration < threshold_durationg))

# softmax 


df_grouped_diags['softmax_index'] = df_grouped_diags.s3_s4_duration.apply(lambda x: int(x/softmax_window) if x < max_predication else int(max_predication/softmax_window))

bit_format = '{0:0' + str(softmax_bits) + 'b}'


if binary_or_softmax == 1:
	df_grouped_diags['bi_label'] = df_grouped_diags.softmax_index.apply(lambda x: [int(d) for d in '{0:03b}'.format(x)])





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


# Filter out diags after start of s3

df_grouped_diags['yearmonth_1'] = df_grouped_diags.min_dt_s3.apply(lambda x: (datetime.datetime.strptime(x, "%Y-%m-%d")-relativedelta.relativedelta(months=1)).strftime("%Y-%m-%d").split('-'))
df_grouped_diags['year_s3_1'] = df_grouped_diags['yearmonth_1'].apply(lambda x:int(x[0]))
df_grouped_diags['month_s3_1'] = df_grouped_diags['yearmonth_1'].apply(lambda x: int(x[1]))

df_grouped_diags['yearmonth'] = df_grouped_diags.min_dt_s3.apply(lambda x: x.split('-'))
df_grouped_diags['year_s3'] = df_grouped_diags['yearmonth'].apply(lambda x: int(x[0]))
df_grouped_diags['month_s3'] = df_grouped_diags['yearmonth'].apply(lambda x: int(x[1]))

df_grouped_diags_s3_history = df_grouped_diags[((df_grouped_diags['year_s3'] == df_grouped_diags['yearno']) & (df_grouped_diags['month_s3']== df_grouped_diags['monthno'])) | ((df_grouped_diags['year_s3_1'] == df_grouped_diags['yearno']) & (df_grouped_diags['month_s3_1']== df_grouped_diags['monthno']))]



df_all_diags_s3_history = df_grouped_diags_s3_history.groupby('dsysrtky',as_index=False).agg({'diags_all':'sum', 'min_dt_s3': 'max', 's3_s4_duration': 'max', 'bi_label': 'max'})


# # # add two columns together 
# # # TODO
# # # remove empty diag 
df_all_diags_s3_history['diags_all_noduplicate'] = df_all_diags_s3_history.diags_all.apply(lambda x: list(filter(None,list(set(x)))))
df_all_diags_s3_history['key_pid_diag'] = df_all_diags_s3_history[['dsysrtky', 'diags_all_noduplicate', 'min_dt_s3']].values.tolist()

# logging.critical(df_grouped_diags.head(10))
# logging.critical(df_all_diags_s3_history.head(10))
# logging.critical(df_all_diags_s3_history.columns.tolist())
# 
# # ----------------------- end of prepare data -----------------------------------------------------------------------


# generate codebook
logging.critical('start codebook')

# logging.critical("ckd related icd9 codes are %s",df_icd9_related_ckd.icd9.tolist())


list_icd9_related_ckd_5853 = df_icd9_related_ckd.dgnscd.tolist()
list_icd9_related_ckd_5854 = df_icd9_related_ckd_5854.dgnscd.tolist()
list_icd9_related_ckd =list(set(list_icd9_related_ckd_5853 + list_icd9_related_ckd_5854))

list_hcpcs_most_freq = df_hcpcs_most_freq_5853.hcpcs_cd.tolist()


logging.critical("ckd related icd9 codes are %s",list_icd9_related_ckd)

list_icd9_related_ckd_max = [s+'_max' for s in list_icd9_related_ckd]
print(list_icd9_related_ckd_max)

if no_ccs == 1:
	list_icd_ccs = list_icd9_related_ckd + list_icd9_related_ckd_max + ['naccs'] + list_hcpcs_most_freq
else:
	list_icd_ccs = list_icd9_related_ckd + list_icd9_related_ckd_max+ df_ccs_binary.ccs.tolist() + ['naccs'] + list_hcpcs_most_freq # add a naccs for the icd that doesn't have and available ccs 

# list_icd_ccs = list_icd9_related_ckd + df_ccs_binary.ccs.tolist() + ['naccs'] + list_icd9_related_ckd_max # add a naccs for the icd that doesn't have and available ccs 
df_codebook = pd.DataFrame({'code':list_icd_ccs})
df_codebook['index_col'] = df_codebook.index



# dict_icd9_ccs = df_icd9_to_ccs.set_index('icd9').T.to_dict('list')

dict_icd9_ccs = dict(zip(df_icd9_to_ccs.icd9, df_icd9_to_ccs.ccs))

dict_codebook = dict(zip(df_codebook.code, df_codebook.index_col))


def days_between(d1, d2):
	d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
	d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
	return abs((d2 - d1).days)

	



def icd_convert(diag_list, dict_codebook,dict_icd9_ccs,dict_pid_diag,dict_pid_hcpcs_count,list_hcpcs_most_freq, pid, min_dt_s3):
	converted_diag = [0] * len(dict_codebook)
	if old_diag == 1:
		for diag_old in diag_list:
			diag = diag_old.replace(" ", "")
			if diag == '[]':
				continue
			if diag in df_codebook.code.tolist():
				code = dict_codebook[diag]
			elif  no_ccs == 0:
				ccs_code = dict_icd9_ccs.get(diag) 
	# 			ccs_code = df_icd9_to_ccs.loc(df_icd9_to_ccs['icd9'] == diag)
				if ccs_code is None:
					code = dict_codebook['naccs']
				else: 
					code = dict_codebook[ccs_code]
			else:
				code = dict_codebook['naccs']
			converted_diag[code] = 1
# 	else:
	if beyond_s3 == 1:
		min_dt_s3_date = datetime.datetime.strptime(min_dt_s3, "%Y-%m-%d")
		center_s3 = min_dt_s3_date-relativedelta.relativedelta(months=3)
		diff_s3 = min_dt_s3_date-relativedelta.relativedelta(months=1)
		upper_s3 = min_dt_s3_date+relativedelta.relativedelta(days=cut_off_ahead)
		for diag in df_codebook.code.tolist():
			diag = str(diag).replace(" ", "")
			if diag in df_codebook.code.tolist(): 
				code = dict_codebook[diag]
				if (str(pid),str(diag)) in dict_diag_thru_dt_list.keys():
					temp_list = dict_diag_thru_dt_list[(str(pid),str(diag))]
					for item in temp_list: 
						item = item.replace(' ','')
						date_item = datetime.datetime.strptime(item, "%Y-%m-%d")
	# 					if (date_item >= center_s3) and (date_item < min_dt_s3_date):
	# 						converted_diag[code] = 1 
						if (date_item >= diff_s3) and (date_item <= upper_s3):
							new_code = dict_codebook[diag + '_max'] 
							converted_diag[new_code] = 1
# 		diag_days_to_s3 = days_between(dict_pid_diag[(str(pid), str(s3_code))], dict_pid_diag[(str(pid), str(diag))]) # only keep window_size record
# 					
# 		diag_days_to_s3 = min(diag_days_to_s3, window_size)
# 		if converted_diag[code] == 0:
# 			converted_diag[code] = diag_days_to_s3
# 		else:
# 			converted_diag[code] = min(diag_days_to_s3,converted_diag[code])
	# update hcpcs
	
	if old_hcpcs == 1:
		for hcpcs in list_hcpcs_most_freq:
			code = dict_codebook[hcpcs]
			if (str(pid),str(hcpcs)) in dict_pid_hcpcs_count.keys():
				converted_diag[code] = dict_pid_hcpcs_count[(str(pid),hcpcs)]
	else:
		min_dt_s3_date = datetime.datetime.strptime(min_dt_s3, "%Y-%m-%d")
		center_s3 = min_dt_s3_date-relativedelta.relativedelta(months=3)
		for hcpcs in list_hcpcs_most_freq:
			code = dict_codebook[hcpcs]
			if (str(pid),str(hcpcs)) in dict_hcpcs_thru_dt_list.keys():
				temp_list = dict_hcpcs_thru_dt_list[(str(pid),hcpcs)]
				for item in temp_list: 
					item = item.replace(' ','')
					date_item = datetime.datetime.strptime(item, "%Y-%m-%d")
					if (date_item > center_s3) and (date_item < min_dt_s3_date):
						converted_diag[code] = converted_diag[code] + 1

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
	

logging.critical('calculate diag_duration')

df_all_diags_s3_history['diag_duration'] =  df_all_diags_s3_history.key_pid_diag.apply(lambda x: icd_convert(x[1],dict_codebook,dict_icd9_ccs,dict_pid_diag,dict_pid_hcpcs_count,list_hcpcs_most_freq,int(x[0]), x[2]))

# durationarray = np.array(df_all_diags_s3_history[['diag_duration']].tolist())
# print(durationarray.shape())
# print(durationarray[:,-1].sort())


logging.critical(df_all_diags_s3_history.columns.to_list())
logging.critical(df_all_diags_s3_history[['diags_all_noduplicate', 'key_pid_diag', 'diag_duration']].head(10))

			
save_csv = 1
if save_csv ==1 : 
	df_input = df_all_diags_s3_history['diag_duration']
	df_output = df_all_diags_s3_history[['bi_label', 's3_s4_duration']] 


	array_input = np.array(df_input.values.tolist()) 
	array_output = np.array(df_output.values.tolist())

	array_csv = np.concatenate((array_input, array_output), axis=1)

	pd.DataFrame(array_csv).to_csv("/home/ywang86/csv/df_all.csv", header=None)


plt.switch_backend('agg')

distrbution = df_output['s3_s4_duration'].values.tolist()

plt.figure(3, figsize=(18, 18))

plt.hist(distrbution, bins = 40, range=[0, 80])
plt.title('total data s3-s4 duration distribution')
plt.savefig('./plots/distribution.png')



			
logging.critical("start training")

tr, ts = train_test_split(df_all_diags_s3_history)

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

print("the shape of array_s1 is ", array_s1.shape)
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
('classification', MLPClassifier(activation='relu', alpha=1e-05, batch_size='lbfgs',beta_1=0.9, beta_2=0.999, early_stopping=False,epsilon=1e-08, hidden_layer_sizes=(7,6), learning_rate='constant',learning_rate_init=0.001, max_iter=10000, momentum=0.9,nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,warm_start=False))
])

clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='lbfgs',beta_1=0.9, beta_2=0.999, early_stopping=False,epsilon=1e-08, hidden_layer_sizes=(7,4), learning_rate='constant',learning_rate_init=0.001, max_iter=10000, momentum=0.9,nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,warm_start=False)


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
	
	count_2d = np.zeros((int(max_predication/softmax_window+1), int(max_predication/softmax_window+1)))
	
	x_post = range(int(max_predication/softmax_window+1))
	for item in b:
		if abs(item[0]) < 2:
			count_correct[item[2]]=count_correct[item[2]]+1
		count_all[item[2]]=count_all[item[2]]+1
		count_2d[item[2]][item[1]] = count_2d[item[2]][item[1]] + 1

	for i in range(len(count_correct)):
		count_rate[i] = count_correct[i]/count_all[i]
	
	print("count_correct is %s ", count_correct)
	print("count_all is %s ",count_all)
	print("count_2d is%s" , count_2d )
	
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
