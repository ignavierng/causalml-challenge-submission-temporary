#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
TODO: 1. first see the visualize.zip and know why we want to do such feature engineering
      2. see line134, continue kci tests for dataset3&4 on private (might take several hours)
      3. check ChaLearn's code for more possible features (see line 185, 299)
      
'''

import os, random, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import euclidean, minkowski, cosine, jaccard

def _get_stat(a_list_of_values, percentiles=(5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)):
    if len(a_list_of_values) == 0: a_list_of_values = [0.]
    return np.append([np.min(a_list_of_values), np.max(a_list_of_values),
                      np.mean(a_list_of_values), np.std(a_list_of_values)],
                     np.percentile(a_list_of_values, percentiles))

def _get_pairwise_stat(arr1, arr2):
    return np.hstack([  # maybe more hypothesis tests. check ChaLearn's code
        tuple(pearsonr(arr1, arr2)),
        tuple(spearmanr(arr1, arr2)),
        tuple(kendalltau(arr1, arr2)),
        [euclidean(arr1, arr2)],
        [cosine(arr1, arr2)],
        [jaccard(arr1, arr2)],
    ])

TRANSLATE = {
    'self_gain': {
        'self_gain': ('data_undo_gains', 1, None,),
        'self_last_value': ('data_wzdo', 1, -1,),
        'parent_last_value': ('data_wzdo', 1, -1,),
        'parent_last_gain': ('data_wzdo_gains', None, -1,),
    },
    'diff_gain': {
        'diff_gain': ('data_undo_gains_remove_self', 1, None,),
        'self_last_value': ('data_wzdo', 1, -1,),
        'parent_last_value': ('data_wzdo', 1, -1,),
        'parent_last_gain': ('data_wzdo_gains', None, -1,),
    },
    'self_value': {
        'self_value': ('data_only_remove_do', 2, None,),
        'self_last_value': ('data_wzdo', 1, -1,),
        'parent_last_value': ('data_wzdo', 1, -1,),
        'parent_last_gain': ('data_wzdo_gains', None, -1,),
    },
}


def _inv_sigmoid(y):
    y[y <= 0] = 1e-15
    y[y >= 1] = 1 - 1e-15
    return np.log(y / (1 - y))

def preprocess_data(dataset_name, dataset_id):
    data_path = './data/Task_1_data_{}_csv/dataset_{}/train.csv'.format(dataset_name, dataset_id)

    ##### Load data #####
    original_data = pd.read_csv(data_path, header=None).to_numpy()  # (#students * #time_steps, 2 + #constructs)
    num_constructs = original_data.shape[1] - 2
    student_ids = np.unique(original_data[:, 0]).astype(int)

    ##### all these are dictionaries with lookup dict[student_id][construct_id], returns an 1-d array #####
    do_indexs = {sid: {} for sid in student_ids}  # in shape (k,), k is different for each construct (usually ~8); value in range [0, #time_steps - 2] (we don't count the last do)
    data_raw = {sid: {} for sid in student_ids}  # in shape (time_steps,): data before inverse sigmoid, in [0, 1]
    data_wzdo = {sid: {} for sid in student_ids}  # in shape (time_steps,): data with do actions (after inverse sigmoid)
    data_wzdo_gains = {sid: {} for sid in student_ids}  # in shape (time_steps-1,): where g[i] = data_wzdo[i+1] - data_wzdo[i]: comes from [do actions]+[self-slow-gains]+[parent-last-gain]
    data_do_pure_gains = {sid: {} for sid in student_ids}  # array in length k, respective to the k timesteps in do_indexs[sid][cid]
    data_undo = {sid: {} for sid in student_ids}  # in shape (time_steps,): data without do actions (remove do-add on self)
    data_undo_gains = {sid: {} for sid in student_ids}  # in shape (time_steps-1,): where g[i] = data_undo[i+1] - data_undo[i]: comes from [self-slow-gains]+[parent-last-gain]
    data_undo_gains_remove_self = {sid: {} for sid in student_ids}  # then how to further remove the self-gain? 1. estimate by f(self-value), or 2. estimate by mean of left and right
    data_only_remove_do = {sid: {} for sid in student_ids}  # in shape (time_steps,): to find the relationship between xt-1 and xt (not that useful)

    ##### Fill in data into lookups #####
    def _gains_remove_environment(series_data, indexs):
        # define the environment as the mean data of the before and after index
        environment_values_of_indexs = []
        original_values_of_indexs = series_data[indexs]
        for timei in indexs:
            beforei, afteri = timei, timei
            while beforei in indexs: beforei -= 1  # e.g., find the last time before timei which is normal "env" data (not do)
            while afteri in indexs: afteri += 1
            env_values = []
            env_weights = []  # defined as 1/(distance to timei)
            if beforei >= 0: env_values.append(series_data[beforei]); env_weights.append(1 / (timei - beforei))
            if afteri < len(series_data): env_values.append(series_data[afteri]); env_weights.append(1 / (afteri - timei))
            environment_values_of_indexs.append(np.average(env_values, weights=env_weights))
        return original_values_of_indexs - np.array(environment_values_of_indexs)

    def _gains_remove_environment_self(series_gain):
        window_size = 2
        kernel = [0.5 / window_size] * window_size + [0] + [0.5 / window_size] * window_size
        padded_pure_gains = np.pad(series_gain, window_size, 'edge')
        env_gains = np.convolve(padded_pure_gains, kernel, mode='valid')
        return series_gain - env_gains

    for sid in student_ids:
        s_actions = original_data[original_data[:, 0] == sid][:, 1][:-1]  # we don't care the last action
        s_data_raw = original_data[original_data[:, 0] == sid][:, 2:]
        s_data = _inv_sigmoid(s_data_raw)  # preprocess
        for cid in range(num_constructs):
            do_indexs[sid][cid] = np.where(s_actions == cid)[0]
            data_raw[sid][cid] = s_data_raw[:, cid]
            data_wzdo[sid][cid] = s_data[:, cid]
            data_wzdo_gains[sid][cid] = np.diff(s_data[:, cid])  # with time lag only one
            data_do_pure_gains[sid][cid] = _gains_remove_environment(data_wzdo_gains[sid][cid], do_indexs[sid][cid])
            data_undo_gains[sid][cid] = np.copy(data_wzdo_gains[sid][cid])
            data_undo_gains[sid][cid][do_indexs[sid][cid]] -= data_do_pure_gains[sid][cid]
            data_undo[sid][cid] = np.append([0], np.cumsum(data_undo_gains[sid][cid])) + data_wzdo[sid][cid][0]
            data_only_remove_do[sid][cid] = np.copy(data_wzdo[sid][cid])
            data_only_remove_do[sid][cid][do_indexs[sid][cid] + 1] -= data_do_pure_gains[sid][cid]
            data_undo_gains_remove_self[sid][cid] = _gains_remove_environment_self(data_undo_gains[sid][cid])

    return {'do_indexs': do_indexs, 'data_raw': data_raw, 'data_wzdo': data_wzdo, 'data_wzdo_gains': data_wzdo_gains,
            'data_do_pure_gains': data_do_pure_gains, 'data_undo': data_undo, 'data_undo_gains': data_undo_gains,
            'data_only_remove_do': data_only_remove_do, 'data_undo_gains_remove_self': data_undo_gains_remove_self}

def get_feature_importance_on_pretrained_models(dataset_name, dataset_id, include_cond_independence=False):
    '''
    since we try to fit as X_{t+1} = f(X_t, PA_{t} - PA_{t-1}), for each variable we train a regression model to predict X_{t+1}
        from X_t and other variables' last time gain, i.e., Y_{t} - Y_{t-1}. Note that here X and Y's respective variables are
        stacked from all students.
        Here we choose Y from two kinds of constructs:
            1) all the other 49 constructs
            2) X's 'ancestor' constructs, where 'ancestors' are estimated by the thresholding high recall result.
        for details of the regression models training and feature importances,
            see ./saved_importances/*produce_regress*. and maybe you can also try other training methods.
        to view such regression's accuracy, see from the downloaded visualize.zip, visualize/saved_importances/*/*/*/*/scatter_*.png

    in addition to regression model, we also tried to test the conditional independence of each X_{t+1} and Y_{t} - Y_{t-1}
        given X_t. Also see visualize/plots/distinguish_features/conditional_independence for its performance.
    TODO: samplesize is 2000. see ./saved_importances/*produce_conditional_independence*.py for details.
          there're still some variables in private/dataset3 and thw whole private/dataset4 not tested.
          it's better to test them and produce results first. (may take several hours for these two datasets).
          so that we can use include_cond_independence=True in this function.
    '''
    PFNMS = ['all_others', 'possible_ancestors']
    MTHDS = ['xgboost', 'lightgbm']
    importances_ratio_of_i_on_j = {j: {i: [] for i in range(50) if i != j} for j in range(50)}
    for pfname in PFNMS:
        for method in MTHDS:
            for j in range(50):
                pth = f'./saved_importances/Task_1_data_{dataset_name}_csv/dataset_{dataset_id}/predict_self_gain_on_self_value_and_{pfname}_last_gain/{method}/feature_importances_{j}.json'
                with open(pth, 'r') as f: importances = json.load(f)
                train_r2, test_r2 = importances['train_r2'], importances['test_r2']
                importances = {k: v for k, v in importances.items() if k not in ["train_r2", "test_r2"]} # skip the r2 results
                sum = np.sum([v for k, v in importances.items() if k.startswith('parent_last_gain')]) # sum of importances of others' last gain
                num_of_features = len(importances)
                importances_ranks = {key: rank / num_of_features for rank, key in enumerate(sorted(importances, key=importances.get, reverse=True))}
                # ranks: the smaller the better. in range [0, 1] (1. denotes not in predict_for); actually already sorted when saving
                for i in range(50):
                    if i == j: continue
                    key = f'parent_last_gain-{i}'
                    value = importances[key] / sum if key in importances else 0.
                    rank = importances_ranks[key] if key in importances_ranks else 1.
                    importances_ratio_of_i_on_j[j][i].extend([value, rank])
    if include_cond_independence:
        for j in range(50):
            pth = f'./saved_importances/Task_1_data_{dataset_name}_csv/dataset_{dataset_id}/condind/conditional_independences_{j}.json'
            with open(pth, 'r') as f: cond_independences = json.load(f)
            for i in range(50):
                if i == j: continue
                importances_ratio_of_i_on_j[j][i].extend(
                    [cond_independences[f'{i}_fisherz_p'],
                     cond_independences[f'{i}_kci_p'],
                     cond_independences[f'{i}_kci_s']])
    # each cell is: ['all_others_xgboost_imp_value', 'all_others_xgboost_imp_rank', 'all_others_lightgbm_imp_value', 'all_others_lightgbm_imp_rank',
    #       'possible_ancestors_xgboost_imp_value', 'possible_ancestors_xgboost_imp_rank', 'possible_ancestors_lightgbm_imp_value', 'possible_ancestors_lightgbm_imp_rank',
    #       'condind_fisherz_p', 'condind_kci_p', 'condind_kci_s']
    return importances_ratio_of_i_on_j

def compute_features_for_stacked_students(dataset_name, dataset_id):
    '''
    i.e., for each (i, j) pair in a dataset, there is only one feature (one row) - calculated from all students.
    in this function you'll see those magic numbers, e.g., PERCENTAGE, 'c_undo_gains': [(0.8, 0.9, 0.95), ...
    what are they? please see visualize/plots/pairwise_scatters/self_gain_on_others_last_gain or diff_gain_on_others_last_gain
    we want to capture the idea that, if p->c is a direct parent, then when p's last gain is high, there should be more number
    of c's this gain (or diff gain) that are high in value. so we use the percentage of c's this gain (or diff gain) that are
    "high" (i.e., in the range of PERCENTAGE) to represent the relationship between p and c.
    TODO: 1. this is quite rough. we could first see visualize/plots/pairwise_scatters and try some better ways to capture the idea.
          2. in addition to our current features, we could have a look at http://clopinet.com/isabelle/Projects/NIPS2013/
             the winner codes, e.g., ProtoML, and see if we can use some of their features (so many pairwise features... e.g., pearsonr..)
             their methods can be applied to our data of either (X_t; Y_t) or (X_gain_t; Y_gain_t-1) or ...
    '''
    PERCENTAGE = np.array([0.03, 0.05, 0.07, 0.1, 0.2, 0.3])
    RANKOFBIGGEST = np.array([4, 5, 6, 7, 8])

    def _reaching_percentile_timestamps(time_series_data):
        start_value, end_value = np.mean(time_series_data[:3]), np.mean(time_series_data[-3:])
        values_thresholds = start_value + (end_value - start_value) * PERCENTAGE
        return np.argmax(time_series_data[:, None] >= values_thresholds, axis=0)

    def _biggest_values_indexs_of_array(array):
        return np.argsort(array)[::-1][RANKOFBIGGEST]

    dres = preprocess_data(dataset_name, dataset_id)
    feature_npy_path = f'./saved_features/features_{dataset_name}_{dataset_id}_stacked_students.npy'
    # if os.path.exists(feature_npy_path): return np.load(feature_npy_path)

    importances_ratio_of_i_on_j = get_feature_importance_on_pretrained_models(dataset_name, dataset_id)

    undo_reaching_timestamps = {
        i: np.array(
            [_reaching_percentile_timestamps(dres['data_undo'][sid][i]) for sid in dres['do_indexs'].keys()])
        for i in range(50)
    }
    wzdo_biggest_jump_indexs = {
        i: np.array([_biggest_values_indexs_of_array(dres['data_wzdo_gains'][sid][i]) for sid in
                     dres['do_indexs'].keys()])
        for i in range(50)
    }

    ci_to_cj_features = []
    for c in range(50):
        ps = [_ for _ in range(50) if _ != c]
        for p in ps:
            onefeature = [p, c]  # p for parent, c for child
            onefeature.extend(importances_ratio_of_i_on_j[c][p])
            feature_names = ['all_others_xgboost_imp_value', 'all_others_xgboost_imp_rank',
                             'all_others_lightgbm_imp_value', 'all_others_lightgbm_imp_rank',
                             'possible_ancestors_xgboost_imp_value', 'possible_ancestors_xgboost_imp_rank',
                             'possible_ancestors_lightgbm_imp_value', 'possible_ancestors_lightgbm_imp_rank',
                             # 'condind_fisherz_p', 'condind_kci_p', 'condind_kci_s'
                             ]

            # OFF COURSE, you can add more features here (e.g., also cond independence in regression; diff-gains;...)
            c_undo_gains = []
            c_diff_gains = []
            p_last_gains = []
            for sid in dres['do_indexs'].keys():
                pred_name, from_ind, to_ind = TRANSLATE['self_gain']['self_gain']
                c_undo_gains.extend(dres[pred_name][sid][c][from_ind:to_ind])
                pred_name, from_ind, to_ind = TRANSLATE['diff_gain']['diff_gain']
                c_diff_gains.extend(dres[pred_name][sid][c][from_ind:to_ind])
                pred_name, from_ind, to_ind = TRANSLATE['diff_gain']['parent_last_gain']
                p_last_gains.extend(dres[pred_name][sid][p][from_ind:to_ind])
            c_undo_gains, c_diff_gains, p_last_gains = np.array(c_undo_gains), np.array(c_diff_gains), np.array(p_last_gains)
            ranges = [(-np.inf, 0.), (0, 0.17), (0.17, 1.0), (1.0, 1.75), (1.75, np.inf)]
            indexs = [np.where((p_last_gains >= r[0]) & (p_last_gains < r[1]))[0] for r in ranges]
            thresholds = {
                'c_undo_gains': [
                    (0.8, 0.9, 0.95),
                    (0.3, 0.4, 0.5),
                    (0.005, 0.01, 0.015),
                    (0.005, 0.01),
                    (-0.01, 0.0, 0.05, 0.07, 0.09, 0.1, 0.13, 0.15),
                ],
                'c_diff_gains': [
                    (0, 0.1),
                    (0.1,),
                    (0.005, 0.01, 0.02),
                    (0.02,),
                    (0, 0.05, 0.1)
                ]
            }
            two_c_gains = {'c_undo_gains': c_undo_gains, 'c_diff_gains': c_diff_gains}
            for gname, cgain in two_c_gains.items():
                for ri, (rleft, rright) in enumerate(ranges):
                    yvalues = cgain[indexs[ri]]
                    for trhd in thresholds[gname][ri]:
                        perc = (yvalues < trhd).mean()
                        onefeature.append(perc)
                        feature_names.append(f'{gname}_{rleft:.2f}-{rright:.2f}_{trhd:.3f}')

            reaching_timestamps_diff_undo_undo = undo_reaching_timestamps[c] - undo_reaching_timestamps[p]
            percentiles = [40, 50, 60]
            stats = np.vstack([np.mean(reaching_timestamps_diff_undo_undo, axis=0),
                       np.percentile(reaching_timestamps_diff_undo_undo, percentiles, axis=0)])
            onefeature.extend(stats.flatten())
            feature_names.extend([f'reaching_timestamps_diff_undo_undo_{int(reach*100)}_perc_{perc}' for perc in ['mean'] + percentiles for reach in PERCENTAGE])

            biggest_jump_indexs_diff_wzdo_wzdo = wzdo_biggest_jump_indexs[c] - wzdo_biggest_jump_indexs[p]
            percentiles = [40, 50, 60]
            stats = np.vstack([np.mean(biggest_jump_indexs_diff_wzdo_wzdo, axis=0),
                               np.percentile(biggest_jump_indexs_diff_wzdo_wzdo, percentiles, axis=0)])
            onefeature.extend(stats.flatten())
            feature_names.extend([f'biggest_jump_indexs_diff_wzdo_wzdo_{rank+1}th_perc_{perc}' for perc in ['mean'] + percentiles for rank in RANKOFBIGGEST])

            ci_to_cj_features.append(onefeature)

    features = np.vstack(ci_to_cj_features)
    print(features.shape)
    np.save(feature_npy_path, features)
    # write feature names to json
    feature_names_path = feature_npy_path.replace('features_', 'feature_names_').replace('.npy', '.json')
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)

def compute_features_for_each_single_students(dataset_name, dataset_id):
    '''
    this is very similar to compute_features_for_stacked_students, but now for each (i, j) pair, there are 100 features (rows),
    i.e., each student has a feature for a row. Then, these single-student features are used for first stage of training.
    TODO: 1. e.g., the 'get_pairwise_stat' can also be applied to stacked students
          2. plot the features found for each stduent and see whether they are reasonable
          3. still, find more features (with insight from ChaLearn 2013, with our targeted formulation,
                or even go beyond it - e.g., parents last value to self this value...)
    '''
    PERCENTAGE = np.array([0.03, 0.05, 0.07, 0.1, 0.2, 0.3])
    RANKOFBIGGEST = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    def _reaching_percentile_timestamps(time_series_data):
        start_value, end_value = np.mean(time_series_data[:3]), np.mean(time_series_data[-3:])
        values_thresholds = start_value + (end_value - start_value) * PERCENTAGE
        return np.argmax(time_series_data[:, None] >= values_thresholds, axis=0)

    def _biggest_values_indexs_of_array(array):
        return np.argsort(array)[::-1][RANKOFBIGGEST]

    dres = preprocess_data(dataset_name, dataset_id)

    feature_npy_path = f'./saved_features/features_{dataset_name}_{dataset_id}_each_single_students.npy'

    undo_reaching_timestamps = {
        i: {sid: _reaching_percentile_timestamps(dres['data_undo'][sid][i]) for sid in dres['do_indexs'].keys()}
        for i in range(50)
    }

    wzdo_reaching_timestamps = {
        i: {sid: _reaching_percentile_timestamps(dres['data_wzdo'][sid][i]) for sid in dres['do_indexs'].keys()}
        for i in range(50)
    }

    undo_biggest_jump_indexs = {
        i: {sid: _biggest_values_indexs_of_array(dres['data_undo_gains'][sid][i]) for sid in dres['do_indexs'].keys()}
        for i in range(50)
    }

    wzdo_biggest_jump_indexs = {
        i: {sid: _biggest_values_indexs_of_array(dres['data_wzdo_gains'][sid][i]) for sid in dres['do_indexs'].keys()}
        for i in range(50)
    }

    ci_to_cj_features = []
    num_of_timesteps = 400

    for p in range(50):
        cs = [_ for _ in range(50) if _ != p]
        for c in cs:
            for sid in dres['do_indexs'].keys():
                onefeature = [p, c]
                feature_names = []
                pred_name, from_ind, to_ind = TRANSLATE['self_gain']['self_gain']
                c_undo_gains = dres[pred_name][sid][c][from_ind:to_ind]
                pred_name, from_ind, to_ind = TRANSLATE['diff_gain']['diff_gain']
                c_diff_gains = dres[pred_name][sid][c][from_ind:to_ind]
                pred_name, from_ind, to_ind = TRANSLATE['diff_gain']['parent_last_gain']
                p_last_gains = dres[pred_name][sid][p][from_ind:to_ind]
                c_undo_gains, c_diff_gains, p_last_gains = np.array(c_undo_gains), np.array(c_diff_gains), np.array(p_last_gains)
                ranges = [(-np.inf, 0.), (0, 0.17), (0.17, 1.0), (1.0, 1.75), (1.75, np.inf)]
                indexs = [np.where((p_last_gains >= r[0]) & (p_last_gains < r[1]))[0] for r in ranges]
                thresholds = {
                    'c_undo_gains': [
                        (0.8, 0.9, 0.95),
                        (0.3, 0.4, 0.5),
                        (0.005, 0.01, 0.015),
                        (0.005, 0.01),
                        (-0.01, 0.0, 0.05, 0.07, 0.09, 0.1, 0.13, 0.15),
                    ],
                    'c_diff_gains': [
                        (0, 0.1),
                        (0.1,),
                        (0.005, 0.01, 0.02),
                        (0.02,),
                        (0, 0.05, 0.1)
                    ]
                }
                two_c_gains = {'c_undo_gains': c_undo_gains, 'c_diff_gains': c_diff_gains}
                for gname, cgain in two_c_gains.items():
                    for ri, (rleft, rright) in enumerate(ranges):
                        yvalues = cgain[indexs[ri]]  # might be empty??
                        for trhd in thresholds[gname][ri]:
                            perc = (yvalues < trhd).mean()
                            onefeature.append(perc)
                            feature_names.append(f'{gname}_{rleft:.2f}-{rright:.2f}_{trhd:.3f}')

                reaching_timestamps_diff_undo_undo = undo_reaching_timestamps[c][sid] - undo_reaching_timestamps[p][sid]
                onefeature.extend(reaching_timestamps_diff_undo_undo)
                feature_names.extend([f'reaching_timestamps_diff_undo_undo_{int(reach * 100)}' for reach in PERCENTAGE])

                reaching_timestamps_diff_undo_wzdo = undo_reaching_timestamps[c][sid] - wzdo_reaching_timestamps[p][sid]
                onefeature.extend(reaching_timestamps_diff_undo_wzdo)
                feature_names.extend([f'reaching_timestamps_diff_undo_wzdo_{int(reach * 100)}' for reach in PERCENTAGE])

                biggest_jump_indexs_diff_undo_undo = undo_biggest_jump_indexs[c][sid] - undo_biggest_jump_indexs[p][sid]
                onefeature.extend(biggest_jump_indexs_diff_undo_undo)
                feature_names.extend([f'biggest_jump_indexs_diff_undo_undo_{rank + 1}th' for rank in RANKOFBIGGEST])

                biggest_jump_indexs_diff_undo_wzdo = undo_biggest_jump_indexs[c][sid] - wzdo_biggest_jump_indexs[p][sid]
                onefeature.extend(biggest_jump_indexs_diff_undo_wzdo)
                feature_names.extend([f'biggest_jump_indexs_diff_undo_wzdo_{rank + 1}th' for rank in RANKOFBIGGEST])

                biggest_jump_indexs_diff_wzdo_wzdo = wzdo_biggest_jump_indexs[c][sid] - wzdo_biggest_jump_indexs[p][sid]
                onefeature.extend(biggest_jump_indexs_diff_wzdo_wzdo)
                feature_names.extend([f'biggest_jump_indexs_diff_wzdo_wzdo_{rank + 1}th' for rank in RANKOFBIGGEST])

                p_jump_indexs = np.array([_ for _ in wzdo_biggest_jump_indexs[p][sid] if _ < num_of_timesteps - 2]).astype(int)
                p_do_indexs = np.array([_ for _ in dres['do_indexs'][sid][p] if _ < num_of_timesteps - 2]).astype(int)
                c_undo_gains_after_p_jumps = dres['data_undo_gains'][sid][c][p_jump_indexs + 1]
                c_undo_gains_after_p_dos = dres['data_undo_gains'][sid][c][p_do_indexs + 1]
                c_diff_gains_after_p_jumps = dres['data_undo_gains_remove_self'][sid][c][p_jump_indexs + 1]
                c_diff_gains_after_p_dos = dres['data_undo_gains_remove_self'][sid][c][p_do_indexs + 1]
                HIGH_PERCENTILES = [60, 70, 80, 90, 95]
                NAMES_OF_STATS = ['min', 'max', 'mean', 'std']
                onefeature.extend(_get_stat(c_undo_gains_after_p_jumps, percentiles=HIGH_PERCENTILES))
                feature_names.extend([f'c_undo_gains_after_p_jumps_{p}' for p in NAMES_OF_STATS + HIGH_PERCENTILES])
                onefeature.extend(_get_stat(c_undo_gains_after_p_dos, percentiles=HIGH_PERCENTILES))
                feature_names.extend([f'c_undo_gains_after_p_dos_{p}' for p in NAMES_OF_STATS + HIGH_PERCENTILES])
                onefeature.extend(_get_stat(c_diff_gains_after_p_jumps, percentiles=HIGH_PERCENTILES))
                feature_names.extend([f'c_diff_gains_after_p_jumps_{p}' for p in NAMES_OF_STATS + HIGH_PERCENTILES])
                onefeature.extend(_get_stat(c_diff_gains_after_p_dos, percentiles=HIGH_PERCENTILES))
                feature_names.extend([f'c_diff_gains_after_p_dos_{p}' for p in NAMES_OF_STATS + HIGH_PERCENTILES])

                onefeature.extend(_get_pairwise_stat(undo_reaching_timestamps[c][sid], undo_reaching_timestamps[p][sid]))
                onefeature.extend(_get_pairwise_stat(c_undo_gains, p_last_gains))
                onefeature.extend(_get_pairwise_stat(dres['data_undo'][sid][c][1:], dres['data_wzdo'][sid][p][:-1]))
                for nnnm in ['undo_reaching_timestamps', 'c_undo_gains_p_last_gains', 'data_undo_c_data_wzdo_p']:
                    feature_names.extend([f'{nnnm}_{stat}' for stat in ['pearsonr1', 'pearsonr2',
                                                                        'spearmanr1', 'spearmanr2',
                                                                        'kendalltau1', 'kendalltau2',
                                                                        'euclidean', 'cosine', 'jaccard']])
                assert len(feature_names) == len(onefeature) - 2
                ci_to_cj_features.append(onefeature)

    features = np.vstack(ci_to_cj_features)
    print(features.shape)
    np.save(feature_npy_path, features)
    # write feature names to json
    feature_names_path = feature_npy_path.replace('features_', 'feature_names_').replace('.npy', '.json')
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f)

def compute_labels_for_localdevs(dataset_id):
    dag_true_path = f'./data/Task_1_data_local_dev_csv/adj_matrix.npy'
    adj_matrix_true = np.load(dag_true_path).astype(bool)[dataset_id]

    for tag in ['each_single', 'stacked']:
        saved_features = np.load(f'./saved_features/features_local_dev_{dataset_id}_{tag}_students.npy')
        i_to_j_labels = []
        for i, j in saved_features[:, :2].astype(int):
            if adj_matrix_true[i, j]: i_to_j_labels.append(0)
            elif adj_matrix_true[j, i]: i_to_j_labels.append(1)
            else: i_to_j_labels.append(2)
        # 3-class classification: 0 for i->j, 1 for j->i, 2 for no edge
        np.save(f'./saved_features/labels_local_dev_{dataset_id}_{tag}_students.npy', np.array(i_to_j_labels))

if __name__ == '__main__':
    for nm in ['local_dev', 'public', 'private']:
        for did in range(5):
            compute_features_for_each_single_students(nm, did)
            compute_features_for_stacked_students(nm, did)
            if nm == 'local_dev': compute_labels_for_localdevs(did)