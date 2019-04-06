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


import datetime
import logging

logging.basicConfig(filename='prediction_2.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


dir_csv = '/home/ywang86/csv/'
s3_s4_start_csv = 'yw_1115_min_s3_s4_positive_6m.csv'
grouped_diags = 'yw_1115_grouped_diags_withbilabel.csv'
icd9_related_ckd = 'yw_icd9_list_ccs_49_50_156_157_158_161.csv'
icd9_to_ccs = 'icd9_ccs_single_level_dx.csv'
ccs_binary = 'ccs_id_binary.csv'
# record_all_csv =  'yw_1115_record_patients_s3_s4.csv'
pid_diags_dict = 'yw_pid_diags_dict_demo.csv'

thre_ave_pa = 1
s3_code = '5853'


def csv_df(csv_name):
	csv_dir = dir_csv + csv_name 
	df = pd.read_csv(csv_dir)
	return df
# select input features

df_s3_s4_start = csv_df(s3_s4_start_csv)
df_grouped_diags = csv_df(grouped_diags)
df_icd9_related_ckd = csv_df(icd9_related_ckd)
df_icd9_to_ccs = csv_df(icd9_to_ccs)
df_ccs_binary = csv_df(ccs_binary) 
# df_record_all = csv_df(record_all_csv) 
df_pid_diags_dict = csv_df(pid_diags_dict)

logging.critical('total number of patients is ')
logging.critical(df_s3_s4_start.shape[0])


cls = df_s3_s4_start.columns.tolist()

cls = df_grouped_diags.columns.tolist()

logging.critical(df_pid_diags_dict.head(10))



dict_pid_diag = {(x,y):z for [x, y, z] in df_pid_diags_dict[['dsysrtky', 'diag', 'min_thru_dt']].values.tolist()} 

dict_pid_age = {x:y for [x, y] in df_pid_diags_dict[['dsysrtky', 'dob_dt']].values.tolist()} 

# logging.critical(dict_pid_diag)



# logging.critical(df_record_all.head(10))
# logging.critical(df_record_all.columns.tolist())

# df_record_pre_dict = pd.DataFrame(columns = ['dsysrtky', 'diag', 'thru_dt'])
# for i in range(12):
# 	diag_select_str = 'icd_dgns_cd' + str(i+1)
# 	logging.critical(diag_select_str)	
# 	df_record_dict = df_record_all[['dsysrtky', diag_select_str, 'thru_dt']]
# 	df_record_dict.columns = ['dsysrtky', 'diag', 'thru_dt']
# 	df_record_pre_dict = df_record_pre_dict.append(df_record_dict)
# 	logging.critical(df_record_dict.shape)
# logging.critical(df_record_pre_dict.shape)
# logging.critical(df_record_pre_dict.head(10))

# dict_pid_diag = {}
# for row in df_record_pre_dict.iterrows():
# # 	dict_pid_diag[row[0]][row[1]] = min(row[2],dict_pid_diag[row[0]][row[1]])
# 	if [row[0],row[1]] in dict_pid_diag.keys():
# 		dict_pid_diag[row[0]][row[1]] = min(row[2],dict_pid_diag[row[0]][row[1]])
# 	else:
# 		dict_pid_diag[row[0]][row[1]] = row[2]
#     


# ----------------------- prepare data -----------------------------------------------------------------------
avg_duration = df_s3_s4_start['s3_s4_duration'].mean()

logging.critical("average duration from s3 to s4 is %s ",avg_duration)

threshold_durationg = thre_ave_pa * avg_duration
df_grouped_diags = df_grouped_diags.assign(bi_label=(df_grouped_diags.s3_s4_duration < threshold_durationg))
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
df_grouped_diags['yearmonth'] = df_grouped_diags.min_dt_s3.apply(lambda x: x.split('-'))



df_grouped_diags['year_s3'] = df_grouped_diags['yearmonth'].apply(lambda x: int(x[0]))
df_grouped_diags['month_s3'] = df_grouped_diags['yearmonth'].apply(lambda x: int(x[1]))

df_grouped_diags_s3_history = df_grouped_diags[(df_grouped_diags['year_s3'] >= df_grouped_diags['yearno']) & (df_grouped_diags['month_s3']>= df_grouped_diags['monthno'])]



df_all_diags_s3_history = df_grouped_diags_s3_history.groupby('dsysrtky',as_index=False).agg({'diags_all':'sum', 'min_dt_s3': 'max', 's3_s4_duration': 'max', 'bi_label': 'max'})


# # # add two columns together 
# # # TODO
# # # remove empty diag 
df_all_diags_s3_history['diags_all_noduplicate'] = df_all_diags_s3_history.diags_all.apply(lambda x: list(filter(None,list(set(x)))))
df_all_diags_s3_history['key_pid_diag'] = df_all_diags_s3_history[['dsysrtky', 'diags_all_noduplicate']].values.tolist()
df_all_diags_s3_history['age'] = df_all_diags_s3_history.dsysrtky.apply(lambda x: dict_pid_age[x])
print("patient age list is", df_all_diags_s3_history[['dsysrtky','age']])

# logging.critical(df_grouped_diags.head(10))
# logging.critical(df_all_diags_s3_history.head(10))
# logging.critical(df_all_diags_s3_history.columns.tolist())
# 
# # ----------------------- end of prepare data -----------------------------------------------------------------------


# generate codebook
logging.critical('start codebook')

logging.critical("ckd related icd9 codes are %s",df_icd9_related_ckd.icd9.tolist())
list_icd_ccs = df_icd9_related_ckd.icd9.tolist() + df_ccs_binary.ccs.tolist() + ['naccs'] # add a naccs for the icd that doesn't have and available ccs 
df_codebook = pd.DataFrame({'code':list_icd_ccs})
df_codebook['index_col'] = df_codebook.index


df_icd9_to_ccs.set_index('icd9')

# dict_icd9_ccs = df_icd9_to_ccs.set_index('icd9').T.to_dict('list')

dict_icd9_ccs = dict(zip(df_icd9_to_ccs.icd9, df_icd9_to_ccs.ccs))

dict_codebook = dict(zip(df_codebook.code, df_codebook.index_col))


def days_between(d1, d2):
    d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)


def icd_convert(diag_list, dict_codebook,dict_icd9_ccs,dict_pid_diag, pid):
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
		converted_diag[code] = max(days_between(dict_pid_diag[(pid, s3_code)], dict_pid_diag[(pid, diag)]),converted_diag[code])
	return converted_diag
	
# 	
# 	
# # 
logging.critical('calculate diag_duration')

df_all_diags_s3_history['diag_duration'] =  df_all_diags_s3_history.key_pid_diag.apply(lambda x: icd_convert(x[1],dict_codebook,dict_icd9_ccs,dict_pid_diag,int(x[0])))

logging.critical(df_all_diags_s3_history.columns.to_list())
logging.critical(df_all_diags_s3_history[['diags_all_noduplicate', 'key_pid_diag', 'diag_duration']].head(10))
			
			
logging.critical("start training")

tr, ts = train_test_split(df_all_diags_s3_history)


df_s1 = tr['diag_duration']
df_s3 = tr['age']
df_s2 = tr[['bi_label']]

test_s1 = ts['diag_duration']
test_s2 = ts[['bi_label']]
test_s3 = tr['age']
print(df_s3.values.shape)
array_s1 =np.concatenate([np.array(df_s1.values.tolist()), df_s3.values.reshape[13658, 1]], axis=1) 
array_s2 = df_s2.values

print(array_s1.shape)
# array_s1 = df_s1
# array_s2 = df_s2
print(test_s3.values.shape)

test_array_s1 = np.concatenate([np.array(test_s1.values.tolist()), test_s3.values], axis=1)
test_array_s2 = test_s2.values
# 
# test_array_s1 = test_s1
# test_array_s2 = test_s2


logging.critical(array_s1[0:10])
# fit model
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 8), random_state=1)
# clf = MLPClassifier(activation='tanh', alpha=1e-05, beta_1=0.9, beta_2=0.999, early_stopping=False,epsilon=1e-08, hidden_layer_sizes=(7, 4), learning_rate='constant',learning_rate_init=0.001, max_iter=10000, momentum=0.9,nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,warm_start=False)
# clf = RandomForestClassifier()

clf.fit(array_s1, array_s2)   


                      

logging.critical("finished training")


predicted_result = clf.predict(array_s1)
predicted_prob = clf.predict_proba(array_s1)

logging.critical("sum of predition is: %s",sum(predicted_result))
logging.critical("sum of real result is: %s ",sum(array_s2))
logging.critical("total number of sample is: %s ",len(array_s2))


error_num_1 = 0
error_num_0 = 0

for i in range(len(array_s2)):
    if (predicted_result[i]!= array_s2[i]) and (array_s2[i] == 1):
        error_num_1 = error_num_1 + 1
    elif (predicted_result[i]!= array_s2[i]) and (array_s2[i] == 0):
    	error_num_0 = error_num_0 + 1
        #logging.critical("predict",i, " wrong, real case is",array_s2[i]," with prob ", predicted_prob[i])
    #elif predicted_result[i] == 1:
     #   logging.critical("predict", i, " right, real case is",array_s2[i]," with prob ", predicted_prob[i])
accuracy_score = clf.score(array_s1, array_s2, sample_weight=None)
logging.critical("accuracy_score is: %s ",accuracy_score)
logging.critical("1-recall is %s ", error_num_1/len(array_s2))
logging.critical("1-precision is %s ",error_num_0/len(array_s2))  
                           


logging.critical('**********auc analysis***********')

probs = clf.predict_proba(array_s1)
fpr, tpr, thresholds = roc_curve(array_s2, probs[:,1])
auc_rf = auc(fpr,tpr)

logging.critical("print probs ")

logging.critical(fpr) 
logging.critical(tpr)
logging.critical(thresholds)

logging.critical('auc for logistic regression is %s', auc_rf)



logging.critical('******************** TEST PART************************')
predicted_result = clf.predict(test_array_s1)
predicted_prob = clf.predict_proba(test_array_s1)

logging.critical("sum of predition is: %s ",sum(predicted_result))
logging.critical("sum of real result is: %s ",sum(test_array_s2))
logging.critical("total number of sample is: %s ",len(test_array_s2))

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

                           


logging.critical('**********auc analysis***********')

probs = clf.predict_proba(test_array_s1)
fpr, tpr, thresholds = roc_curve(test_array_s2, probs[:,1])
auc_rf = auc(fpr,tpr)

plt.plot(fpr,tpr, 'b')
plt.show()

logging.critical("print probs ")
logging.critical(probs)
logging.critical('auc for logistic regression is %s', auc_rf)