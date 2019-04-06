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

logging.basicConfig(filename='learning.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')


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



ave_duration = 359.99681511174566


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

all_csv = 'df_all.csv'

auc_list = []




def deep_learning(thre_ave): 

	df_all = csv_df(all_csv)

	logging.critical("start training")

	df_all['new_bi'] = df_all.iloc[:,-1].apply(lambda x: x < thre_ave )

	tr, ts = train_test_split(df_all)

	df_s1 = tr.iloc[:,0:-4]
	df_s2 = tr['new_bi']


	# if balance_data ==1: 
	# 	print("resample data to balance negative and positive")
	# 	df_s1_s2 = tr[['diag_duration','bi_label']]
	# 	max_size = df_s1_s2['bi_label'].value_counts().max()
	# 	# oversample imbalanced data:
	# 	lst = [df_s1_s2]
	# 	for class_index, group in df_s1_s2.groupby('bi_label'):
	# 		lst.append(group.sample(int((max_size-len(group))), replace=True))
	# 	df_new = pd.concat(lst)
	# 		
	# 	df_s1 = df_new['diag_duration']
	# 	df_s2 = df_new['bi_label']
	


	test_s1 = ts.iloc[:,0:-4]
	test_s2 = ts['new_bi']



	array_s1 = np.array(df_s1.values.tolist()) 
	print("the shape of array_s1 is %s", array_s1.shape)


	array_s2 = np.array(df_s2.values.tolist())

	# array_s1_s2 = np.concatenate((array_s1,array_s2, axis = 1)


	# array_s1 = np.concatenate((array_s1,np.array(df_add.values.tolist())), axis = 1)

	min_max_scaler = preprocessing.MinMaxScaler()
	array_s1 = min_max_scaler.fit_transform(array_s1)





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
		training_auc = auc_rf
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
		test_auc = auc_rf

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
		plt.title('Training data s3-s4 duration distribution')

		plt.subplot(222)
		plt.hist(b[:,1], bins = int(max_predication/softmax_window+1))
		plt.title('Training data s3-s4 prediction distribution')

		plt.subplot(223)
		plt.hist(b[:,0], bins = int(max_predication/softmax_window+1))
		plt.title('Training data s3-s4 prediction error distribution')

		plt.subplot(224)
		plt.bar(x_post,count_rate)
		plt.title('Training data s3-s4 Accuracy Rate')

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
	return [training_auc, test_auc]
auc_all = []

def get_test_auc(x):
    return np.array(x)[:,1].tolist()
    
for j in range(3):
	print ("*************** round ", j, "***************")
	auc_list = []
	for i in range(8):
		thre_ave =(i+1)/8* ave_duration
		print("***************", i, ": ,", thre_ave,  "***************")
		auc_training_test = deep_learning(thre_ave)
		auc_list.append(auc_training_test)
	test_auc_list = get_test_auc(auc_list)
	auc_all.append(test_auc_list)	
	print(auc_all)
logging.critical("auc_all is")
logging.critical(auc_all)

auc_average = np.mean(np.array(auc_all), axis=0)
print(auc_average)
logging.critical("auc_average is")
logging.critical(auc_average)