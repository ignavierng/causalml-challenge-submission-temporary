import os
import json
import random
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt 

def process_data(data_df, stu_meta, X_imputed, start_from_t, real_obs_thres1, real_obs_thres2, filter_k):
    X = []
    T = []
    Y = []
    B = []
    sample_weight = []

    construct_samples_by_year_dict = {}
    construct_samples_by_year_dict_use_last = {}

    X_imputed_start_idx=0
        
    for student_id in data_df['student_id'].drop_duplicates():
        curr_data_df = data_df[data_df['student_id'] == student_id]
        year = int(stu_meta[stu_meta["UserId"]==student_id]["YearGroup"])
        del curr_data_df['student_id']
        curr_data = curr_data_df.values
        curr_intervention = curr_data[:, 0]
        curr_construct_obs = curr_data[:, 1:1+num_constructs]
        curr_is_real_obs = curr_data[:, 1+num_constructs:1+num_constructs*2]
        
        # examine and fill
        len_curr_construct_obs = len(curr_construct_obs)

        curr_notnan_rate = ((~np.isnan(curr_construct_obs)).sum(1)) / (curr_construct_obs.shape[1])

        #for t in range(curr_construct_obs.shape[0]):
        #    for cc in range(curr_construct_obs.shape[1]):
        #        if not np.isnan(curr_construct_obs[t,cc]):
        #            if abs(curr_construct_obs[t,cc]-X_imputed[X_imputed_start_idx:X_imputed_start_idx+len_curr_construct_obs][t,cc])>0.01:
        #                print("!!!!") 

        curr_construct_obs = X_imputed[X_imputed_start_idx:X_imputed_start_idx+len_curr_construct_obs]
        #if curr_construct_obs.max()>1.01 or curr_construct_obs.min()<0:
        #    print("!!!!!!!!!!!!!!!!!")
        curr_construct_obs = np.clip(curr_construct_obs, 0, 1)
        X_imputed_start_idx+=len_curr_construct_obs
        ###################################

        def filter(curr_construct_obs, k):
            curr_construct_obs_filtered = np.zeros_like(curr_construct_obs)
            curr_construct_obs_filtered[0] = curr_construct_obs[0]
            for i in range(1, curr_construct_obs.shape[0]):
                curr_construct_obs_filtered[i] =  curr_construct_obs[i]*k+curr_construct_obs_filtered[i-1]*(1-k)

            return curr_construct_obs_filtered

        curr_construct_obs = filter(curr_construct_obs, filter_k)

        #def plot(obs, title):
        #    f = plt.figure(title)
        #    plt.plot([i for i in range(len(obs))], obs) 
        #    plt.xlabel('time step') 
        #    plt.ylabel('construct obs') 
        #    plt.title(title) 
        #    plt.show() 
        #plot(curr_construct_obs[:,50], 'before filter')
        #for k in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        #    curr_construct_obs_filtered = filter(curr_construct_obs, k)
        #    plot(curr_construct_obs_filtered[:,50], 'after filter k '+str(k))


        for t in range(start_from_t, curr_data.shape[0] - 1):

            if curr_notnan_rate[t] > real_obs_thres2:
                X.append(curr_construct_obs[t])
                B.append(curr_intervention[t])
                Y.append(curr_construct_obs[t+1, :])
                T.append(year)
                sample_weight.append(1)

            elif curr_notnan_rate[t] > real_obs_thres1:
                X.append(curr_construct_obs[t])
                B.append(curr_intervention[t])
                Y.append(curr_construct_obs[t+1, :])
                T.append(year)
                sample_weight.append(0.5)


        if year in construct_samples_by_year_dict_use_last:
            construct_samples_by_year_dict_use_last[year].append(curr_construct_obs[-1])
        else:
            construct_samples_by_year_dict_use_last[year] = [curr_construct_obs[-1]]

        for t in range(start_from_t, curr_data.shape[0]):
            if year in construct_samples_by_year_dict:
                construct_samples_by_year_dict[year].append(curr_construct_obs[t])
            else:
                construct_samples_by_year_dict[year] = [curr_construct_obs[t]]
            
    X = np.array(X)
    Y = np.array(Y)
    T = np.array(T)
    B = np.array(B)

    #sample_weight = np.array(sample_weight)
    #sample_weight = sample_weight / sample_weight.sum()
    #sample_weight = sample_weight*len(X)

    construct_samples_by_year_dict = {k:np.array(v) for k, v in construct_samples_by_year_dict.items()}
    ave_construct_per_year_dict = {k:np.array(v).mean(0) for k, v in construct_samples_by_year_dict.items()}

    construct_samples_by_year_dict_use_last = {k:np.array(v) for k, v in construct_samples_by_year_dict_use_last.items()}
    ave_construct_per_year_dict_use_last = {k:np.array(v).mean(0) for k, v in construct_samples_by_year_dict_use_last.items()}

    return X, Y, T, B, sample_weight, construct_samples_by_year_dict_use_last, ave_construct_per_year_dict_use_last, construct_samples_by_year_dict, ave_construct_per_year_dict

def train(X, Y, T, B, sample_weight, if_construct_asked_list, boosting_type, n_estimators=100, learning_rate=0.1, max_depth=10):

    regressor_ls = []
    for construct_id in range(len(if_construct_asked_list)):
        print("train for ", construct_id)
        if if_construct_asked_list[construct_id] == 1:
            regressor = lgb.LGBMRegressor(boosting_type=boosting_type, num_leaves=31, max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, verbose=-1)
            
            #regressor.fit(X, Y[:, construct_id], sample_weight, eval_metric='rmse', categorical_feature=[0])
            data = np.concatenate((B.reshape(-1,1), T.reshape(-1,1), X), axis=1)
            regressor.fit(data, Y[:, construct_id], sample_weight, eval_metric='rmse', categorical_feature=[0,1])
        else:
            regressor = None
        regressor_ls.append(regressor)

    return regressor_ls

def get_prediction(regressor, year, intervention, X, effect_idx):
    B = np.ones(len(X))*intervention
    T = np.ones(len(X))*year
    data = np.concatenate((B.reshape(-1,1), T.reshape(-1,1), X), axis=1)
    
    ori_performance = X[:, effect_idx]

    Y = regressor.predict(data)
    Y = np.clip(Y, 0, 1)

    return Y
    #return np.maximum(Y, ori_performance) # non decreasing

def test_and_save(test_data_df, const_map, regressor_ls, output_path, use_ave_profile, construct_samples_by_year_dict, ave_construct_per_year_dict, scale = 1.15):

    cate_list = []
    for index, effect_idx in enumerate(test_data_df['QuestionConstructId']):
        print(f"Computing CATE for {effect_idx}")
        treatment_idx = test_data_df.iloc[index]['TreatmentLessonConstructId']
        control_idx = test_data_df.iloc[index]['ControlLessonConstructId']
        year = test_data_df.iloc[index]['Year']
        print("year ", year)
        # Convert to new construct id
        effect_idx, treatment_idx, control_idx = (
            const_map[effect_idx],
            const_map[treatment_idx],
            const_map[control_idx],
        )
        if use_ave_profile:
            construct_score = ave_construct_per_year_dict[year]
            #print(construct_score)
            # use average construct per year
            score_by_treatment = get_prediction(regressor_ls[effect_idx], year, treatment_idx,  construct_score[np.newaxis, :], effect_idx).item()
            score_by_control =   get_prediction(regressor_ls[effect_idx], year, control_idx,    construct_score[np.newaxis, :], effect_idx).item()
        else:
            # average after scoring
            score_by_treatment = get_prediction(regressor_ls[effect_idx], year, treatment_idx, construct_samples_by_year_dict[year], effect_idx).mean()
            score_by_control =   get_prediction(regressor_ls[effect_idx], year, control_idx,   construct_samples_by_year_dict[year], effect_idx).mean()
        
        cur_cate = score_by_treatment - score_by_control
        cate_list.append(cur_cate)
        print(cur_cate, score_by_treatment, score_by_control)


    cate_estimate = np.array(cate_list)  # [num_queries]
    cate_estimate = scale*cate_estimate
    np.save(output_path, cate_estimate)

if __name__ == "__main__":
    
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

    for i, asked_ori_construct_id in enumerate(np.unique(test_data_df["QuestionConstructId"])):
        #if i>1:
        #    break
        asked_construct_id = const_map[asked_ori_construct_id]
        if_construct_asked_list[asked_construct_id] = 1

    impute_type = "rf_imp" #"rf_imp"
    if impute_type == "ori_imp":
        imputed_file = "original_X_imputed.pickle"
    else:
        imputed_file = "X_imputed.pickle"
    print(imputed_file)
    f=open(imputed_file,'rb')
    X_imputed=pickle.load(f)
    f.close()
    print("X_imputed", X_imputed.shape)

    result_file_name = "cate_estimate.npy"

    #-------------------------------------------------------------------------------------------------------------------------------
    start_from_t = 0
    boosting_type = 'gbdt' # 'goss' 'gbdt' 'rf' 'dart'
    filter_k = 1
    real_obs_thres1_ls = [i/1000.0 for i in range(11, 12)]
    max_depth_ls = [10]
    learning_rate_ls = [0.1] 
    n_estimators_ls = [150]

    for n_estimators in n_estimators_ls:
        for learning_rate in learning_rate_ls:
            for max_depth in max_depth_ls:
                for real_obs_thres1 in real_obs_thres1_ls:
                    real_obs_thres2 = real_obs_thres1 + 5/1000.0
                    X, Y, T, B, sample_weight, construct_samples_by_year_dict_use_last, ave_construct_per_year_dict_use_last, construct_samples_by_year_dict, ave_construct_per_year_dict = process_data(data_df, stu_meta, X_imputed, start_from_t, real_obs_thres1, real_obs_thres2, filter_k)
                    
                    regressor_ls = train(X, Y, T, B, sample_weight, if_construct_asked_list, boosting_type, n_estimators, learning_rate, max_depth)
                    
                    test_and_save(test_data_df, const_map, regressor_ls, result_file_name, use_ave_profile=True, 
                        construct_samples_by_year_dict=construct_samples_by_year_dict_use_last, ave_construct_per_year_dict=ave_construct_per_year_dict_use_last)
