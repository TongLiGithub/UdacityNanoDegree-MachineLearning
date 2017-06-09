#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                # 'salary', 
                 #'deferral_payments', 
                 #'total_payments', 
                 #'loan_advances', 
                 #'bonus', 
                 'bonus_total_ratio',
                 #'restricted_stock_deferred', 
                 #'deferred_income', 
                 'total_stock_value', 
                 #'expenses', 
                 'exercised_stock_options', 
                 #'other', 
                 #'long_term_incentive', 
                 #'restricted_stock', 
                 #'director_fees', 
                 #'from_poi_to_this_person', 
                # 'from_this_person_to_poi', 
                 #'shared_receipt_with_poi',
                 'to_messages', 
                 'from_messages',
                 'from_email_poi_proportion',
                 'to_email_poi_proportion'] 


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_dict.keys()    
### Task 2: Remove outliers
#### Remove total, remove the travel agency in the park
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)

### Task 3: Create new feature(s)
#### create bonus-totalpayments ratio
for k, v in data_dict.iteritems():
    if v['total_payments']=='NaN' or v['bonus'] =='NaN':
        v['bonus_total_ratio']='NaN'
    else:
        v['bonus_total_ratio']=float(v['bonus'])/float(v['total_payments'])

data_poi=[]
for point in data_dict:
    poi = data_dict.get(point)['poi']
    data_poi.append(poi)
  
for i in range(len(data_poi)):
    if data_poi[i]==True:
        data_poi[i]=1
    else:
        data_poi[i]=0


bonus_total_ratio=[]
for point in data_dict:
    bt_ratio = data_dict.get(point)['bonus_total_ratio']
    bonus_total_ratio.append(bt_ratio)

plt.scatter(bonus_total_ratio, data_poi)
plt.title('bonus_total_ratio')




#### create from_email_poi_proportion which indicates the proportion of from_this_person_to_poi in from_messages
for k, v in data_dict.iteritems():
    if v['from_messages']=='NaN' or v['from_this_person_to_poi'] =='NaN':
        v['from_email_poi_proportion']='NaN'
    else:
        v['from_email_poi_proportion']=float(v['from_this_person_to_poi'])/float(v['from_messages'])


from_ratio=[]
for point in data_dict:
    fe_ratio = data_dict.get(point)['from_email_poi_proportion']
    from_ratio.append(fe_ratio)


plt.scatter(from_ratio, data_poi)
plt.title('from_email_ratio')


#### create to_email_poi_proportion which indicates the proportion of from_poi_to_this_person in to_messages
for k, v in data_dict.iteritems():
    if v['to_messages']=='NaN' or v['from_poi_to_this_person'] =='NaN':
        v['to_email_poi_proportion']='NaN'
    else:
        v['to_email_poi_proportion']=float(v['from_poi_to_this_person'])/float(v['to_messages'])


to_ratio=[]
for point in data_dict:
    te_ratio = data_dict.get(point)['to_email_poi_proportion']
    to_ratio.append(te_ratio)

plt.scatter(to_ratio, data_poi)
plt.title('to_email_ratio')


#### examine original features with univaraite plots
data_salary=[]
for point in data_dict:
    salary = data_dict.get(point)['salary']
    data_salary.append(salary)

plt.scatter(data_salary, data_poi)


data_loan=[]
for point in data_dict:
    loan = data_dict.get(point)['loan_advances']
    data_loan.append(loan)
    
plt.scatter(data_loan, data_poi)  
plt.title("loan_advances & poi")  
    

data_bonus=[]
for point in data_dict:
    bonus = data_dict.get(point)['bonus']
    data_bonus.append(bonus)
    
plt.scatter(data_bonus, data_poi)  
plt.title("bonus & poi")  


data_def_pay=[]
for point in data_dict:
    defpay = data_dict.get(point)['deferral_payments']
    data_def_pay.append(defpay)
    
plt.scatter(data_def_pay, data_poi)  
plt.title("deferral_payments & poi") 


data_tol_pay=[]
for point in data_dict:
    tolpay = data_dict.get(point)['total_payments']
    data_tol_pay.append(tolpay)
    
plt.scatter(data_tol_pay, data_poi)  
plt.title("total_payments & poi") 


data_res_sto_def=[]
for point in data_dict:
    ressto = data_dict.get(point)['restricted_stock_deferred']
    data_res_sto_def.append(ressto)
    
plt.scatter(data_res_sto_def, data_poi)  #all pois have nan on this variable
plt.title("restricted_stock_deferred & poi") 


data_def_inc=[]
for point in data_dict:
    definc = data_dict.get(point)['deferred_income']
    data_def_inc.append(definc)
    
plt.scatter(data_def_inc, data_poi)  
plt.title("deferred_income & poi") 


data_tol_sto=[]
for point in data_dict:
    tolsto = data_dict.get(point)['total_stock_value']
    data_tol_sto.append(tolsto)
      
plt.scatter(data_tol_sto, data_poi)  
plt.title("total_stock_value & poi") 


data_exp=[]
for point in data_dict:
    exp = data_dict.get(point)['expenses']
    data_exp.append(exp)
    
plt.scatter(data_exp, data_poi)  
plt.title("expenses & poi") 


data_exe_sto=[]
for point in data_dict:
    exesto = data_dict.get(point)['exercised_stock_options']
    data_exe_sto.append(exesto)
    
plt.scatter(data_exe_sto, data_poi)  
plt.title("exercised_stock_options & poi") 


data_long_incen=[]
for point in data_dict:
    longincen = data_dict.get(point)['long_term_incentive']
    data_long_incen.append(longincen)
    
plt.scatter(data_long_incen, data_poi)  
plt.title("long_term_incentive & poi") 


data_res_sto=[]
for point in data_dict:
    ressto = data_dict.get(point)['restricted_stock']
    data_res_sto.append(ressto)
    
plt.scatter(data_res_sto, data_poi)  
plt.title("restricted_stock & poi") 


data_direc=[]
for point in data_dict:
    direc = data_dict.get(point)['director_fees']
    data_direc.append(direc)

plt.scatter(data_direc, data_poi)   #all pois have nan on this variable
plt.title("director_fees & poi") 


data_frompoi=[]
for point in data_dict:
    frompoi = data_dict.get(point)['from_poi_to_this_person']
    data_frompoi.append(frompoi)
    
plt.scatter(data_frompoi, data_poi)  
plt.title("from_poi_to_this_person & poi") 


data_to=[]
for point in data_dict:
    to = data_dict.get(point)['to_messages']
    data_to.append(to)
        
plt.scatter(data_to, data_poi)  
plt.title("to_messages & poi") 


data_fromme=[]
for point in data_dict:
    fromme = data_dict.get(point)['from_messages']
    data_fromme.append(fromme)
    
    
plt.scatter(data_fromme, data_poi)  
plt.title("from_messages & poi") 


data_topoi=[]
for point in data_dict:
    topoi = data_dict.get(point)['from_this_person_to_poi']
    data_topoi.append(topoi)
    
plt.scatter(data_topoi, data_poi)  
plt.title("from_this_person_to_poi & poi") 


data_shared=[]
for point in data_dict:
    shared = data_dict.get(point)['shared_receipt_with_poi']
    data_shared.append(shared)
    
plt.scatter(data_shared, data_poi)  
plt.title("shared_receipt_with_poi & poi") 






### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest

scaler = MinMaxScaler()
select = SelectKBest()
dtc = DecisionTreeClassifier()


# Using Decision Tree as classifier
estimators = [('feature_selection', select),
              ('dtc', dtc)]

# Create pipeline
pipeline = Pipeline(estimators)

params = dict(feature_selection__k=[4,5,6],
              dtc__criterion=['gini', 'entropy'],
              dtc__max_depth=[None, 1, 2, 3, 4],
              dtc__min_samples_split=[2,3,4,5],
              dtc__class_weight=[None, 'balanced'],
              dtc__random_state=[42])



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

from sklearn.cross_validation import train_test_split    
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit()
    
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid=params, cv=sss, scoring='f1')

grid_search.fit(features_train, labels_train)
prediction = grid_search.predict(features_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy = accuracy_score(prediction, labels_test)

precision = precision_score(prediction, labels_test) 
recall = recall_score(prediction, labels_test) 

grid_search.best_score_




grid_search.best_estimator_

select1=SelectKBest(k=4)
select1.fit(features_train, labels_train)

select1.get_support()
select1.scores_
select1.pvalues_

# Print best estimators
print "\n", "Best parameters are: ", grid_search.best_params_, "\n"
print "\n", "Precision is", precision, "\n"
print "\n", "Recall is", recall, "\n"

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf=grid_search.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list)