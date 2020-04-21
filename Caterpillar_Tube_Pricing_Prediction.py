import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from scipy.stats import skew
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

bill_of_materials = pd.read_csv('./data/bill_of_materials.csv')
components = pd.read_csv('./data/components.csv')
comp_adaptor = pd.read_csv('./data/comp_adaptor.csv')
comp_boss = pd.read_csv('./data/comp_boss.csv')
comp_elbow = pd.read_csv('./data/comp_elbow.csv')
comp_float = pd.read_csv('./data/comp_float.csv')
comp_hfl = pd.read_csv('./data/comp_hfl.csv')
comp_nut = pd.read_csv('./data/comp_nut.csv')
comp_other = pd.read_csv('./data/comp_other.csv')
comp_sleeve = pd.read_csv('./data/comp_sleeve.csv')
comp_straight = pd.read_csv('./data/comp_straight.csv')
comp_tee = pd.read_csv('./data/comp_tee.csv')
comp_threaded = pd.read_csv('./data/comp_threaded.csv')
specs = pd.read_csv('./data/specs.csv')
tube = pd.read_csv('./data/tube.csv')
tube_end_form = pd.read_csv('./data/tube_end_form.csv')
type_component = pd.read_csv('./data/type_component.csv')
type_connection = pd.read_csv('./data/type_connection.csv')
type_end_form = pd.read_csv('./data/type_end_form.csv')
train = pd.read_csv('./data/train_set.csv', parse_dates=[2,])
test = pd.read_csv('./data/test_set.csv', parse_dates=[3,])


# =============================================================================
# EDA
# =============================================================================

# outlier detection through IQR proximity rule as the distribution is skewed
def IQR(dataset):    
    for i in dataset.columns:
        if dataset[i].dtype != object:
            IQR = dataset[i].quantile(0.75)-dataset[i].quantile(0.25)
            upper_threshold = dataset[i].quantile(0.75)+3*IQR
            
            if dataset[i][dataset[i] > upper_threshold].any() == True:
                print('\nOutlier: ', i)
                print("Upper Threshold = %0.4f" % upper_threshold)
                print(dataset.loc[dataset[i] == dataset[i].max()])
            

# check columns id (pseudo constant), description, maximum value of None (9999), and NaN value for each table

# (21198, 17)
bill_of_materials.head()
bill_of_materials.isnull().sum()

# (25, 20)
comp_adaptor.head()      
comp_adaptor.isnull().sum()

# C-1868 does not have overall_length, so use length_1 + length_2 = 93.5
comp_adaptor.loc[comp_adaptor['overall_length'].isnull(), 'overall_length'] = comp_adaptor.loc[comp_adaptor['overall_length'].isnull(), 'length_1'] +  comp_adaptor.loc[comp_adaptor['overall_length'].isnull(), 'length_2'] 
 
# C-0443 and C-1695 do not have weight
comp_adaptor.drop(comp_adaptor.index[[8, 21]], inplace=True)

# 'adaptor_angle', 'length_1', 'length_2' only have one value 
# 'unique_feature', 'orientation' are categorical variables which only 1 row contains different value
# 'component_type_id', 'end_form_id_1', 'connection_type_id_1', 'end_form_id_2', 'connection_type_id_2' are not used in other tables(CSV)
comp_adaptor.drop(columns = ['adaptor_angle', 'length_1', 'length_2', 'unique_feature', 'orientation', 'component_type_id', 'end_form_id_1', 'connection_type_id_1', 
                   'end_form_id_2', 'connection_type_id_2'], inplace=True)


# (147, 15)
comp_boss.head()
comp_boss.isnull().sum()
comp_boss = comp_boss[['component_id', 'height_over_tube', 'weight']]

comp_boss.hist(figsize=(20,20))
plt.show()

sns.boxplot(x=comp_boss["height_over_tube"])
sns.boxplot(x=comp_boss["weight"])

IQR(comp_boss)
comp_boss.drop(comp_boss.index[31], inplace=True)


# (6, 9)
comp_hfl.head()
comp_hfl.isnull().sum()
comp_hfl = comp_hfl[['component_id', 'hose_diameter', 'weight']]


# (178, 16)
comp_elbow.head()
comp_elbow.isnull().sum()
comp_elbow.drop(['component_type_id', 'mj_class_code', 'mj_plug_class_code', 'plug_diameter', 
                 'groove', 'unique_feature', 'orientation',], axis=1, inplace=True)

comp_elbow.hist(figsize=(20,20))
plt.show()

IQR(comp_elbow)
comp_elbow.drop(comp_elbow.index[52], inplace=True)


# (16, 7)
comp_float.head()
comp_float.isnull().sum()
comp_float.drop(['component_type_id', 'orientation'], axis=1, inplace=True)


# (65, 11)
comp_nut.head()
comp_nut.isnull().sum()
comp_nut.drop(['component_type_id', 'seat_angle', 'diameter', 'blind_hole', 'orientation'], axis=1, inplace=True)

comp_nut.hist(figsize=(20,20))
plt.show()

IQR(comp_nut)


# (1001, 3)
comp_other.head()
comp_other.drop(['part_name'], axis=1, inplace=True)


# (50, 10)
comp_sleeve.head()
comp_sleeve.drop(['component_type_id', 'connection_type_id', 'unique_feature', 'plating', 'orientation'], axis=1, inplace=True)

comp_sleeve.hist(figsize=(20,20))
plt.show()

IQR(comp_sleeve)
comp_sleeve.drop(comp_sleeve.index[[28, 29, 30, 31, 32, 33, 34, 48]], inplace=True)


# (361, 12)
comp_straight.head()
comp_straight.drop(['component_type_id', 'overall_length', 'mj_class_code', 'head_diameter', 
                    'unique_feature', 'groove', 'orientation'], axis=1, inplace=True)

comp_straight.hist(figsize=(20,20))
plt.show()

IQR(comp_straight)


# (4, 14)
comp_tee.head()
comp_tee.drop(['component_type_id', 'mj_class_code', 'mj_plug_class_code', 'groove', 
               'unique_feature', 'orientation'], axis=1, inplace=True)

comp_tee.hist(figsize=(20,20))
plt.show()

IQR(comp_tee)


# (194, 32)
comp_threaded.head()
comp_threaded.drop(['component_type_id', 'adaptor_angle', 'end_form_id_1', 'connection_type_id_1', 'end_form_id_2',
                    'connection_type_id_2', 'end_form_id_3', 'connection_type_id_3', 'end_form_id_4', 'connection_type_id_4',
                    'nominal_size_4', 'unique_feature', 'orientation'], axis=1, inplace=True)

# There are five columns with length, so I fill NA with 0, summarize length and drop excessive columns
comp_threaded['length_1'] = comp_threaded['length_1'].fillna(0)
comp_threaded['length_2'] = comp_threaded['length_2'].fillna(0)
comp_threaded['length_3'] = comp_threaded['length_3'].fillna(0)
comp_threaded['length_4'] = comp_threaded['length_4'].fillna(0)
comp_threaded['overall_length'] = comp_threaded['overall_length'].fillna(0)
comp_threaded['overall_length'] = comp_threaded['overall_length'] + comp_threaded['length_1'] + comp_threaded['length_2'] + comp_threaded['length_3'] + comp_threaded['length_4']

comp_threaded.drop(['length_1', 'length_2', 'length_3', 'length_4'], axis=1, inplace=True)


comp_threaded.hist(figsize=(20,20))
plt.show()

IQR(comp_threaded)
comp_threaded.drop(comp_threaded.index[[40, 90]], inplace=True)


# (21198, 16)
tube.head()
tube.drop(['material_id', 'end_a_1x', 'end_a_2x', 'end_x_1x', 'end_x_2x', 'end_a',
          'end_x', 'num_boss', 'num_bracket', 'other'], axis=1, inplace=True)

tube.hist(figsize=(20,20))
plt.show()

IQR(tube)
tube.drop(tube.index[[17689,17689,18002,18003,15132, 15174, 15175, 17688, 17689, 18002, 18003, 19320]], inplace=True)

# below  files contain only has text descriptions, so I decided not to use them:

# tube_end_form, type_component, type_connection, type_end_form, components


# =============================================================================
# Feature Engineering
# =============================================================================

train.head()


# create several features from dates for additional information
train['year'] = train.quote_date.dt.year
train['month'] = train.quote_date.dt.month
train['dayofyear'] = train.quote_date.dt.dayofyear
train['dayofweek'] = train.quote_date.dt.dayofweek
train['day'] = train.quote_date.dt.day

test['year'] = test.quote_date.dt.year
test['month'] = test.quote_date.dt.month
test['dayofyear'] = test.quote_date.dt.dayofyear
test['dayofweek'] = test.quote_date.dt.dayofweek
test['day'] = test.quote_date.dt.day

train = train.drop('quote_date',axis=1)
test = test.drop('quote_date',axis=1)


# combine all files with info on components in one file.
all_comp = pd.concat([comp_adaptor, comp_boss, comp_elbow, comp_float, comp_hfl, comp_nut, comp_other,
                      comp_sleeve, comp_straight, comp_tee, comp_threaded])

# (2033, 29)
all_comp.head()
all_comp.isnull().sum()

all_comp = all_comp[['component_id', 'weight', 'length', 'overall_length', 'thickness']]

# combine two length columns.
all_comp['overall_length'] = all_comp['overall_length'].fillna(0)
all_comp['length'] = all_comp['length'].fillna(0)
all_comp['length'] = all_comp['length'] + all_comp['overall_length']
all_comp = all_comp.drop(['overall_length'], axis=1)

all_comp['weight'] = all_comp['weight'].fillna(0)
all_comp['thickness'] = all_comp['thickness'].fillna(0)

# add information about tube itself and the list of components to main files.
train = pd.merge(train, tube, on='tube_assembly_id', how='left')
train = pd.merge(train, bill_of_materials, on ='tube_assembly_id', how='left')
print(train.shape)
test = pd.merge(test, tube, on='tube_assembly_id', how='left')
test = pd.merge(test, bill_of_materials, on ='tube_assembly_id', how='left')
print(test.shape)

# rename columns so that they will be different from length of components.
train.rename(columns={'length': 'length_t'}, inplace = True)
test.rename(columns={'length': 'length_t'}, inplace = True)

# merging to get information about components
for i in range(1, 9, 2):
    suffix1 = '_' + str(i)
    suffix2 = '_' + str(i + 1)
    component_1 = 'component_id' + suffix1
    component_2 = 'component_id' + suffix2
    
    train = pd.merge(train, all_comp, left_on = component_1, right_on = 'component_id', how='left')
    train = pd.merge(train, all_comp, left_on = component_2, right_on = 'component_id', suffixes=(suffix1, suffix2), how='left')
    
    test = pd.merge(test, all_comp, left_on = component_1, right_on = 'component_id', how='left')
    test = pd.merge(test, all_comp, left_on = component_2, right_on = 'component_id', suffixes=(suffix1, suffix2), how='left')

print(train.shape)
train.isnull().sum()

# drop unnecessary columns
train.drop(['component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5', 'component_id_6',
            'component_id_7', 'component_id_8'], axis=1, inplace=True)
test.drop(['component_id_1', 'component_id_2', 'component_id_3', 'component_id_4', 'component_id_5', 'component_id_6',
            'component_id_7', 'component_id_8'], axis=1, inplace=True)

# add descriptive information about specs.
train = pd.merge(train, specs, on='tube_assembly_id', how='left')
test = pd.merge(test, specs, on='tube_assembly_id', how='left')

# calculate various additional features on physical parameters. They turned out to be useful.
length_columns = [col for col in list(train.columns) if 'length' in col]
weight_columns = [col for col in list(train.columns) if 'weight' in col]
thickness_columns = [col for col in list(train.columns) if 'thickness' in col]
train['avg_w'] = train[weight_columns].mean(axis=1)
train['avg_l'] = train[length_columns].mean(axis=1)
train['avg_th'] = train[thickness_columns].mean(axis=1)
train['min_w'] = train[weight_columns].min(axis=1)
train['min_l'] = train[length_columns].min(axis=1)
train['min_th'] = train[thickness_columns].min(axis=1)
train['max_w'] = train[weight_columns].max(axis=1)
train['max_l'] = train[length_columns].max(axis=1)
train['max_th'] = train[thickness_columns].max(axis=1)
test['avg_w'] = test[weight_columns].mean(axis=1)
test['avg_l'] = test[length_columns].mean(axis=1)
test['avg_th'] = test[thickness_columns].mean(axis=1)
test['min_w'] = test[weight_columns].min(axis=1)
test['min_l'] = test[length_columns].min(axis=1)
test['min_th'] = test[thickness_columns].min(axis=1)
test['max_w'] = test[weight_columns].max(axis=1)
test['max_l'] = test[length_columns].max(axis=1)
test['max_th'] = test[thickness_columns].max(axis=1)
train['tot_w'] = train[weight_columns].sum(axis=1)
train['tot_l'] = train[length_columns].sum(axis=1)
test['tot_w'] = test[weight_columns].sum(axis=1)
test['tot_l'] = test[length_columns].sum(axis=1)

feat_name= [col for col in train.columns if train[col].dtype != 'object']
train[feat_name].hist(figsize=(25,25))
plt.show()

# take log of skewered columns to smooth them and fill NA.
for col in train.columns:
    if train[col].dtype != 'object':
        if skew(train[col]) > 0.75:
            train[col] = np.log1p(train[col])
            train[col] = train[col].apply(lambda x: 0 if x == -np.inf else x)

        train[col] = train[col].fillna(0)
        
for col in test.columns:
    if test[col].dtype != 'object':
        if skew(test[col]) > 0.75:
            test[col] = np.log1p(test[col])
            test[col] = test[col].apply(lambda x: 0 if x == -np.inf else x)

        test[col] = test[col].fillna(0)

feat_name= [col for col in train.columns if train[col].dtype != 'object']
train[feat_name].hist(figsize=(25,25))
plt.show()

for col in train.columns:
    if train[col].dtype == 'object':
        train[col].replace(np.nan,' ', regex=True, inplace= True)
for col in test.columns:
    if test[col].dtype == 'object':
        test[col].replace(np.nan,' ', regex=True, inplace= True)
        
X_train = train.drop('cost',axis=1)
Y_train = train['cost']
X_test  = test.drop('id', axis=1)

feat_name = [col for col in X_train.columns if X_train[col].dtypes == 'O']
feat_index = [X_train.columns.get_loc(c) for c in feat_name if c in X_train]

# convert to arrays for easier transformation
X_train = np.array(X_train)
X_test = np.array(X_test)


#label encoding

for i in feat_index:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(X_train[:,i]) + list(X_test[:,i]))
    X_train[:,i] = lbl.transform(X_train[:,i])
    X_test[:,i] = lbl.transform(X_test[:,i])

#XGB need float.
X_train = X_train.astype(float)
X_test = X_test.astype(float)


# =============================================================================
# Model Building and Evaluation
# =============================================================================

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.20, random_state=10)
print('Training Data Size :',x_train.shape)
print('Test Data Size :',x_test.shape)
print('Training Label Size :',y_train.shape)
print('Testing Label Size :',y_test.shape)

param ={
            'n_estimators': [100,500, 1000,1500],
            'max_depth':[2,4,6,8]
        }

xgboost_tree = xgb.XGBRegressor(
    eta = 0.1,
    min_child_weight = 2,
    subsample = 0.8,
    colsample_bytree = 0.8,
    tree_method = 'exact',
    reg_alpha = 0.05,
    silent = 0,
    random_state = 1023
)

grid = GridSearchCV(estimator=xgboost_tree,param_grid=param,cv=5,  verbose=10, 
                    n_jobs=-1,scoring='neg_mean_squared_error')
    
grid_result = grid.fit(x_train, y_train)
best_params = grid_result.best_params_

print('Best Params :',best_params)

from math import sqrt
from sklearn.metrics import mean_squared_error
pred = grid_result.predict(x_test)
print('Root Mean squared error {}'.format(sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))))

# Target: 5% over budget
# Try feature engineering
# Try other model, NN, CatBoost, LightGBM

diff = pd.concat([np.exp(y_test).reset_index(drop='True'),pd.DataFrame(np.exp(pred)).reset_index(drop='True')],axis=1)
diff.columns = ['Orginal','Prediction']
diff['Diff'] = diff.Orginal - diff.Prediction
diff.head()