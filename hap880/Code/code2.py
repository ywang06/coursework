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
