import os
import json
import random
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
from missingpy import MissForest
import time


random.seed(1)
np.random.seed(1)

const_map_path = "../task_4_data_processed/const_map.pickle"
f=open(const_map_path,'rb')
const_map=pickle.load(f)
f.close()
num_constructs = len(const_map)

data_path = '../task_4_data_processed/train_raw.csv'
data_df = pd.read_csv(data_path, header=None)
print("data_df shape: ", data_df.shape)
columns = ['student_id', 'bot_action'] + ['construct_{}'.format(construct_id) for construct_id in range(num_constructs)] + ['obs_is_real_{}'.format(construct_id) for construct_id in range(num_constructs)]
data_df.columns = columns

stu_meta_path = '../task_4_data_raw/student_metadata.csv'
stu_meta = pd.read_csv(stu_meta_path, index_col=False)

test_data_path = "../task_4_data_raw/construct_experiments_input_test.csv"
test_data_df = pd.read_csv(test_data_path)

# find models that are needed for test
if_construct_asked_list = [0 for i in range(num_constructs)]
for asked_ori_construct_id in np.unique(test_data_df["QuestionConstructId"]):
    asked_construct_id = const_map[asked_ori_construct_id]
    if_construct_asked_list[asked_construct_id] = 1

#====================================================================================

# imputation
X_with_nan = []
for student_id in data_df['student_id'].drop_duplicates():
    curr_data_df = data_df[data_df['student_id'] == student_id]
    del curr_data_df['student_id']
    curr_data = curr_data_df.values
    curr_construct_obs = curr_data[:, 1:1+num_constructs]
 
    for t in range(curr_data.shape[0]):
        X_with_nan.append(curr_construct_obs[t])
        if  np.all(np.isnan(curr_construct_obs[t])):
            print(student_id, t)
        
X_with_nan = np.array(X_with_nan)
print(X_with_nan.shape)

# some constructs have no value across all students and all time steps
for j in range(X_with_nan.shape[1]):
    if (X_with_nan.shape[0] - np.isnan(X_with_nan[:,j]).sum()) == 0 :
        print(j)
        X_with_nan[:,j]=0.5

forest_imputer = MissForest()
t1 = time.time()
X_imputed = forest_imputer.fit_transform(X_with_nan[:,:])
t2 = time.time()
print('time:', t2-t1)
print("saving")
f=open(os.path.join("./", "X_imputed.pickle"),'wb')
pickle.dump(X_imputed, f)
f.close()

