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
                 'salary_bonus_ratio',
                 'bonus_total_ratio',
                 #'restricted_stock_deferred', 
                 #'deferred_income', 
                 'total_stock_value', 
                 #'expenses', 
                 'exercised_stock_options', 
                 #'other', 
                 #'long_term_incentive', 
                 #'restricted_stock', 
                 'director_fees', #dramatically increases recall , but precision and acc are much lower
                 #'from_poi_to_this_person', 
                # 'from_this_person_to_poi', 
                 #'shared_receipt_with_poi',
                 'to_messages', 
                 'from_messages',
                 'stock_ratio',
                 'from_ratio',
                 'to_ratio'] 


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_dict.keys()    
### Task 2: Remove outliers
#### Remove total, remove the reavel agency in the park
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


#### create bonus salary ratio
for k, v in data_dict.iteritems():
    if v['bonus']=='NaN' or v['salary'] =='NaN':
        v['salary_bonus_ratio']='NaN'
    else:
        v['salary_bonus_ratio']=float(v['salary'])/float(v['bonus'])


salary_bonus_ratio=[]
for point in data_dict:
    sb_ratio = data_dict.get(point)['salary_bonus_ratio']
    salary_bonus_ratio.append(sb_ratio)

plt.scatter(salary_bonus_ratio, data_poi)
plt.title('salary_bonus_ratio')



#### create exercised-total stock ratio
for k, v in data_dict.iteritems():
    if v['exercised_stock_options']=='NaN' or v['total_stock_value'] =='NaN':
        v['stock_ratio']='NaN'
    else:
        v['stock_ratio']=float(v['exercised_stock_options'])/float(v['total_stock_value'])


stock_ratio=[]
for point in data_dict:
    st_ratio = data_dict.get(point)['stock_ratio']
    stock_ratio.append(st_ratio)

plt.scatter(stock_ratio, data_poi)
plt.title('ex_stock_ratio')



#### create proportion of from_this_person_to_poi as a percentage in from_messages
for k, v in data_dict.iteritems():
    if v['from_messages']=='NaN' or v['from_this_person_to_poi'] =='NaN':
        v['from_ratio']='NaN'
    else:
        v['from_ratio']=float(v['from_this_person_to_poi'])/float(v['from_messages'])


from_ratio=[]
for point in data_dict:
    fe_ratio = data_dict.get(point)['from_ratio']
    from_ratio.append(fe_ratio)

plt.scatter(from_ratio, data_poi)
plt.title('from_email_ratio')


#### create proportion of from_poi_to_this_person as a percentage in to_messages
for k, v in data_dict.iteritems():
    if v['to_messages']=='NaN' or v['from_poi_to_this_person'] =='NaN':
        v['to_ratio']='NaN'
    else:
        v['to_ratio']=float(v['from_poi_to_this_person'])/float(v['to_messages'])


to_ratio=[]
for point in data_dict:
    te_ratio = data_dict.get(point)['to_ratio']
    to_ratio.append(te_ratio)

plt.scatter(to_ratio, data_poi)
plt.title('to_email_ratio')



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




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf=grid_search.best_estimator_
dump_classifier_and_data(clf, my_dataset, features_list)