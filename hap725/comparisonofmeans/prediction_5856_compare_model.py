
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import NearestNeighbors
import scipy.stats as stats
from sklearn import preprocessing

from sklearn.neural_network import MLPClassifier
import random
import logging

import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)




# In[5]:


FIRST_ROW_NUM = 0  # or 0

start_num = 0
num_samples = 8000000
balance = 1
training_rate = 0.8


def generate_input(start_num, num_samples, balance, csv_name):
    #  Create Training Data Set
    high_use_threshold = 50
#     dir_csv = '/Users/yingwang/Google Drive/Study2017/GeorgeMasonUniversity/Summer2018/HAP780/Project/GMUDataBase/code/'
    dir_csv = '/home/ywang86/csv/'

#     file_name = yw_temp_activity_level6_denom_20110401.csv
    csv_dir = dir_csv + csv_name
    with open(csv_dir, 'rt') as infile, open(dir_csv +'partial.csv', 'wt') as outfile:
        outfile.writelines(row for row_num, row in enumerate(infile, FIRST_ROW_NUM)
                            if (((row_num < start_num + num_samples) and (row_num > start_num) ) or (row_num == 0)))
    df_all = pd.read_csv(open(dir_csv +'partial.csv','r',errors='ignore'),skip_blank_lines=True, error_bad_lines=False,low_memory=False)
    df_all = df_all.dropna()


    df_all = df_all.apply(pd.to_numeric, errors='coerce')
    
    df_all = df_all.sample(frac=1)
    query_str_positive = "positive_5856 >= 1"
    query_str_negative = "positive_5856 < 1" 

    if balance:
        df_passed = df_all.query(query_str_positive)
        df_failed = df_all.query(query_str_negative)

        df_failed_truncate = df_failed.head(df_passed.shape[0]*balance) 
        
    
        frames = [df_passed, df_failed_truncate]
        df_all = pd.concat(frames)  
    
        logging.debug("number of positive is %d", df_passed.shape[0])
        logging.debug("num of negative before truncate is %d", df_failed.shape[0])
        logging.debug("num of negative is %d", df_failed_truncate.shape[0])
        logging.debug("total number is %d", df_all.shape[0])


    df_s2 = df_all[['positive_5856']] 

    df_s1 = df_all[[ 'l514','l4439','l5849','l7931','l25050','lv700','l71945','l586','l7840','l2724','l56400','l79902','l4659','l42789','l28521','l4241','l4589','l5939','l25001','l6826','l7802','l5180','lv7283','l7862','l27651','l78907','l5990','l2767','l41400','l2689','l99673','l5855','l78701','l5932','l2859','l72981','l7823','l1101','l78720','lv5882','lv7281','l2449','lv7644','l41401','l4011','l4010','l2382','l7245','l4019','l36616','l25040','l0389','l78659','l7804','l72252','l7242','l43491','l40391','l43310','l5852','l36511','lv5881','l78060','l78609','l7020','l7906','l5859','l42731','lv7612','l2809','l4254','lv420','l2113','l4660','l71946','l5789','l79319','l78009','l2720','lv5869','l25060','l2722','l73390','l78650','l40390','l5853','l7295','l53081','l51881','l78097','l25000','l3804','l78791','l78909','l71941','l4293','l79431','l4279','l4240','l5854','l486','l5845','l4280','l78900','l496','lv5861','l5119','l78079','l51889','l7231','lv0481','l25002','l78959','l78605','sex','race','age','orec','crec','l86803', 'l90960', 'l87186', 'l77001', 'l96372', 'l80053', 'l99221', 'l99214', 'l76700', 'l92015', 'l78452', 'l17000', 'l11721', 'la0427', 'l84460', 'lg0202', 'l83718', 'l99222', 'l84156', 'l82728', 'l83970', 'l1036f', 'l99204', 'l74000', 'l70450', 'l71250', 'l84165', 'l76770', 'l82306', 'l77052', 'l99291', 'l84153', 'l80069', 'l83036', 'lq2038', 'l81002', 'l99205', 'l92250', 'l84443', 'l85025', 'l82570', 'l74176', 'l93000', 'l84439', 'la0425', 'l99309', 'l80061', 'l82607', 'l99284', 'l83540', 'l76705', 'l88305', 'l99306', 'l92004', 'la0429', 'lg0008', 'l90935', 'lq2037', 'l93010', 'l83550', 'lg8427', 'l99223', 'lg8553', 'l81001', 'l76775', 'l93880', 'l99239', 'l90961', 'l93971', 'l36558', 'l84100', 'l99215', 'la0428', 'l99285', 'l99203', 'l3120f', 'l99238', 'l71020', 'l82043', 'l99283', 'l4048f', 'l99233', 'l92014', 'l81003', 'l81000', 'l92012', 'l36556', 'l77080', 'l85027', 'l36415', 'l82746', 'l92134', 'l87086', 'l99308', 'l99232', 'l99212', 'l20610', 'l43239', 'l83735', 'l99213', 'l99231', 'l93970', 'l93306', 'l99211', 'l84550', 'l71010', 'l76937', 'lg8447', 'l99202', 'l80048', 'l82947', 'l6045f', 'l85610']]
#     df_s1 = df_all[[ 'l514','l4439','l5849','l7931','l25050','lv700','l71945','l586','l7840','l2724','l56400','l79902','l4659','l42789','l28521','l4241','l4589','l5939','l25001','l6826','l7802','l5180','lv7283','l7862','l27651','l78907','l5990','l2767','l41400','l2689','l99673','l5855','l78701','l5932','l2859','l72981','l7823','l1101','l78720','lv5882','lv7281','l2449','lv7644','l41401','l4011','l4010','l2382','l7245','l4019','l36616','l25040','l0389','l78659','l7804','l72252','l7242','l43491','l40391','l43310','l5852','l36511','lv5881','l78060','l78609','l7020','l7906','l5859','l42731','lv7612','l2809','l4254','lv420','l2113','l4660','l71946','l5789','l79319','l78009','l2720','lv5869','l25060','l2722','l73390','l78650','l40390','l5853','l7295','l53081','l51881','l78097','l25000','l3804','l78791','l78909','l71941','l4293','l79431','l4279','l4240','l5854','l486','l5845','l4280','l78900','l496','lv5861','l5119','l78079','l51889','l7231','lv0481','l25002','l78959','l78605','state_cd','cnty_cd','sex','race','age','orec','crec']]


    array_s1 = df_s1.values
    array_s2 = df_s2.values


    return array_s1, array_s2    


csv_file_list = ['yw_temp_activity_level6_denom_20110401.csv', 'yw_temp_activity_level6_denom_20110701.csv', 'yw_temp_activity_level6_denom_20111001.csv', 'yw_temp_activity_level6_denom_20120101.csv','yw_temp_activity_level6_denom_20120401.csv', 'yw_temp_activity_level6_denom_20120701.csv']

# csv_file_list = ['yw_temp_activity_level6_denom_20110401.csv']
for i in range(len(csv_file_list)):
    array_s1, array_s2 = generate_input(start_num, num_samples, balance, csv_file_list[i])
   
    if i == 0:
        array_s1_all = array_s1
        array_s2_all = array_s2
    else:
        array_s1_all = np.concatenate((array_s1_all, array_s1), axis=0)
        array_s2_all = np.concatenate((array_s2_all, array_s2), axis=0)
    logging.debug(np.shape(array_s1_all))


s12_all = np.concatenate((array_s1_all, array_s2_all), axis=1)
np.random.shuffle(s12_all) 

# np.savetxt("/home/ywang86/csv/balancedfile.csv", s12_all, delimiter=",")

df = pd.DataFrame(array_s2_all)
df.to_csv("/home/ywang86/csv/newbalancedfile.csv")


training_num = int(len(s12_all)*training_rate)
s1_training = s12_all[0:training_num,0:-1]
s2_training = s12_all[0:training_num,-1]

s1_test = s12_all[training_num:,0:-1]
s2_test = s12_all[training_num:,-1]




logging.debug("start training")

# fit model
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 8), random_state=1)
clf = MLPClassifier(activation='tanh', alpha=1e-05, batch_size='lbfgs',beta_1=0.9, beta_2=0.999, early_stopping=False,epsilon=1e-08, hidden_layer_sizes=(7, 4), learning_rate='constant',learning_rate_init=0.001, max_iter=10000, momentum=0.9,nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,warm_start=False)
clf.fit(s1_training, s2_training)                         

logging.debug("finished training")



predicted_result = clf.predict(s1_training)
predicted_prob = clf.predict_proba(s1_training)


# logging.debug("%%%%%%%%%%%%%%%%Training Accuracy%%%%%%%%%%%%%%%%")
# logging.debug("sum of predition is: ",sum(predicted_result))
# logging.debug("sum of real result is: ",sum(s2_training))

error_num = 0
for i in range(len(s2_training)):
    if (predicted_result[i]!= s2_training[i]) and (s2_training[i] == 1):
        error_num = error_num + 1
        #logging.debug("predict",i, " wrong, real case is",array_s2[i]," with prob ", predicted_prob[i])
    #elif predicted_result[i] == 1:
     #   logging.debug("predict", i, " right, real case is",array_s2[i]," with prob ", predicted_prob[i])
accuracy_score = clf.score(s1_training, s2_training, sample_weight=None)
# logging.debug("accuracy_score is: ",accuracy_score)
# logging.debug("1-recall is", error_num/len(s2_training))  
# logging.debug("high use rate is", sum(s2_training)/len(s2_training))
                           
logging.debug("%%%%%%%%%%%%%%%%Testing Accuracy%%%%%%%%%%%%%%%%")

                       
# Testing Result
test_start_num = 80000
test_num_samples = 100000



test_result = clf.predict(s1_test)                           
logging.debug("sum of test_prediction is: %d",sum(test_result))
logging.debug("sum of real test is: %d",sum(s2_test)) 
error_num = 0
false_positive = 0
false_negative = 0
true_positive = 0
true_negative = 0
for i in range(len(s2_test)):
    if (test_result[i]!= s2_test[i]):
        error_num = error_num + 1
        if s2_test[i] == 1:
            false_negative = false_negative + 1
        else:
            false_positive = false_positive + 1  
    else:
        if s2_test[i] == 1:
            true_positive = true_positive + 1
        else:
            true_negative = true_negative + 1
        
logging.debug("total number is %d", len(s1_test))
logging.debug("false positive is .4f", false_positive, "false positive rate is ", false_positive/len(s1_test))
logging.debug("false negative is .4f", false_negative, "false negative rate is ", false_negative/len(s1_test))
logging.debug("true positive is .4f", true_positive, "true positive rate is ", true_positive/len(s1_test))
logging.debug("true negative is .4f", true_negative, "true negative rate is ", true_negative/len(s1_test))
     
accuracy_score = clf.score(s1_test, s2_test, sample_weight=None)


logging.debug("test accuracy_score is: %.4f",accuracy_score)
recall = true_positive/(true_positive+false_negative)
logging.debug("recall is  %.4f", recall)
precision = true_positive/(false_positive+true_positive)
logging.debug("precision is  %.4f", precision)

logging.debug("%%%%%%%%%%%%%%%%Training Accuracy%%%%%%%%%%%%%%%%")




test_result = clf.predict(s1_training)                           
logging.debug("sum of test_prediction is: %d",sum(test_result))
logging.debug("sum of real test is: %d",sum(s2_training)) 
error_num = 0
false_positive = 0
false_negative = 0
true_positive = 0
true_negative = 0
for i in range(len(s2_training)):
    if (test_result[i]!= s2_training[i]):
        error_num = error_num + 1
        if s2_training[i] == 1:
            false_negative = false_negative + 1
        else:
            false_positive = false_positive + 1  
    else:
        if s2_training[i] == 1:
            true_positive = true_positive + 1
        else:
            true_negative = true_negative + 1
        
logging.debug("total number is %d", len(s1_training))
logging.debug("false positive is %.4f", false_positive, "false positive rate is ", false_positive/len(s1_training))
logging.debug("false negative is .4f", false_negative, "false negative rate is ", false_negative/len(s1_training))
logging.debug("true positive is .4f", true_positive, "true positive rate is ", true_positive/len(s1_training))
logging.debug("true negative is .4f", true_negative, "true negative rate is ", true_negative/len(s1_training))
     
accuracy_score = clf.score(s1_training, s2_training, sample_weight=None)

logging.debug("Training accuracy_score is:  %.4f",accuracy_score)

recall = true_positive/(true_positive+false_negative)
logging.debug("recall is  %.4f", recall)
precision = true_positive/(false_positive+true_positive)
logging.debug("precision is  %.4f", precision)


from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


# Learn to predict each class against the other
X_train = s1_training;
y_train = s2_training;

y_test = s2_test;

X_test = s1_test;
y_score = clf.predict_proba(X_test)


precision, recall, thresholds = precision_recall_curve(y_test, y_score[:,1])
pr_auc = auc(recall, precision)
print("pr_auc is: ", pr_auc)
logging.debug("pr_auc is  %.4f", pr_auc)

logging.debug(recall, precision, thresholds)


fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
ax.plot(recall, precision)
plt.xlabel('recall')
plt.ylabel('precision') 
plt.title('AUC plot for ESRD')
fig.savefig('/home/ywang86/csv/auc.png')   # save the figure to file
plt.close(fig) 



fpr, tpr, thresholds = roc_curve(y_test,  y_score[:,1])

print(fpr,tpr)
fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
lw = 2
ax.plot(fpr, tpr,lw=lw, label='ROC curve (area = %0.2f)' % pr_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('Receiver operating characteristic for ESRD Prediction')
plt.legend(loc="lower right")
fig.savefig('/home/ywang86/csv/roc.png')   # save the figure to file
plt.close(fig) 


print('FINISHED')


