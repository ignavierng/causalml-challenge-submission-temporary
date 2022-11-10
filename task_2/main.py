import json
import random

import lightgbm as lgb
import numpy as np
import pandas as pd


def inv_sigmoid(array):
    array[array == 0] = 1e-15
    array[array == 1] = 1 - 1e-15
    return np.log(array / (1 - array))


def sigmoid(array):
    return 1 / (1 + np.exp(-array))


def get_single_prediction(regressor_ls, curr_X, curr_target, adj_mapping, adj_matrix):
    curr_X[:50] = inv_sigmoid(curr_X[:50])
    curr_X_transform = transform_single_data(curr_X, adj_mapping, curr_target)
    parents = list(np.where(adj_matrix[:, curr_target])[0])
    features = [curr_target] + parents + [50]
    curr_X_transform = curr_X_transform[features]
    curr_Y_predicted = regressor_ls[curr_target].predict(curr_X_transform[np.newaxis, :]).item()
    curr_Y_predicted = sigmoid(curr_Y_predicted)
    # Clip the value at interval [0, 1]
    if curr_Y_predicted < 0:
        curr_Y_predicted = 0
    if curr_Y_predicted > 1:
        curr_Y_predicted = 1
    return curr_Y_predicted


def transform_data(X, adj_mapping, construct_id):
    num_constructs = len(adj_mapping)
    X_transform = X.copy()
    parents_mapping = adj_mapping[:, construct_id]
    parents_mapping_dict = dict(zip(range(num_constructs), parents_mapping))
    mapping_func = lambda cid: parents_mapping_dict[cid]
    X_transform[:, -1] = np.array(list(map(mapping_func, X_transform[:, -1])))
    return X_transform

def transform_single_data(curr_X, adj_mapping, construct_id):
    X = curr_X[np.newaxis, :]
    return transform_data(X, adj_mapping, construct_id)[0]


def create_adj_mapping(adj_matrix):
    # adj_mapping will be later used to map bot_action to new categories
    # self: 0
    # non_parents: 1
    # parents: start from 2
    num_constructs = len(adj_matrix)
    adj_mapping = np.zeros_like(adj_matrix)
    for construct_id in range(num_constructs):
        parents = list(np.where(adj_matrix[:, construct_id])[0])
        non_parents = list(set(np.where(1 - adj_matrix[:, construct_id])[0]) - {construct_id})
        adj_mapping[construct_id, construct_id] = 0
        adj_mapping[non_parents, construct_id] = 1
        parent_mapping = 2
        for parent in parents:
            adj_mapping[parent, construct_id] = parent_mapping
            parent_mapping += 1
    return adj_mapping


def main():
    num_constructs = 50
    all_cate_predicted = []
    for dataset_id in range(5):
        random.seed(1)
        np.random.seed(1)
        data_path = '../../../../data/Task_1_data_private_csv/dataset_{}/train.csv'.format(dataset_id)
        data_df = pd.read_csv(data_path, header=None)
        columns = ['student_id', 'bot_action'] + ['construct_{}'.format(construct_id) for construct_id in range(num_constructs)]
        data_df.columns = columns

        interventions_path = '../../../../data/Task_2_data_private/intervention_{}.json'.format(dataset_id)
        with open(interventions_path) as json_file:
            interventions = json.load(json_file)

        # Load the estimated adjacency matrix on private dataset
        adj_matrix_predicted = np.load('adj_matrix_private.npy')[dataset_id]
        adj_mapping = create_adj_mapping(adj_matrix_predicted)

        X_ls = []
        Y_ls = []
        # Add historical data from task 1 for training models
        for student_id in data_df['student_id'].drop_duplicates():
            curr_data_df = data_df[data_df['student_id'] == student_id]
            del curr_data_df['student_id']
            curr_data = curr_data_df.values
            for t in range(curr_data.shape[0] - 3):
                curr_X = curr_data[t, 1:].reshape(-1)
                curr_X = np.append(curr_X, curr_data[t + 1, 0])
                curr_Y = curr_data[t + 3, 1:]
                X_ls.append(curr_X)
                Y_ls.append(curr_Y)

        # Add historical data from task 2 for training models
        for intervention in interventions:
            curr_data = np.array(intervention['conditioning'])
            for t in range(curr_data.shape[0] - 3):
                curr_X = curr_data[t, 1:].reshape(-1)
                curr_X = np.append(curr_X, curr_data[t + 1, 0])
                curr_Y = curr_data[t + 3, 1:]
                X_ls.append(curr_X.reshape(-1))
                Y_ls.append(curr_Y)

        X = np.array(X_ls)
        Y = np.array(Y_ls)
        # Pre-process the data
        X[:, :50] = inv_sigmoid(X[:, :50])
        Y = inv_sigmoid(Y)

        # Train models
        regressor_ls = []
        for construct_id in range(num_constructs):
            parents = list(np.where(adj_matrix_predicted[:, construct_id])[0])
            features = [construct_id] + parents + [50]
            X_transform = transform_data(X, adj_mapping, construct_id)
            X_transform = X_transform[:, features]    # Select only parental features
            regressor = lgb.LGBMRegressor(boosting_type='goss', num_leaves=47, max_depth=12, verbose=-1)
            regressor.fit(X_transform, Y[:, construct_id], eval_metric='rmse',
                          categorical_feature=[len(features) - 1])
            regressor_ls.append(regressor)

        cate_predicted = []
        for intervention in interventions:
            curr_data = np.array(intervention['conditioning'])
            t = len(curr_data) - 1
            curr_X = curr_data[t, 1:].reshape(-1)
            curr_target = np.where(intervention['effect_mask'])[1].item() - 1

            # Intervention
            intervened_bot_action = int(intervention['intervention'][0][0])
            intervened_curr_X = np.append(curr_X, intervened_bot_action)
            intervention_prediction = get_single_prediction(regressor_ls, intervened_curr_X, curr_target, adj_mapping, adj_matrix_predicted)

            # Reference
            reference_bot_action = int(intervention['reference'][0][0])
            reference_curr_X = np.append(curr_X, reference_bot_action)
            reference_predidction = get_single_prediction(regressor_ls, reference_curr_X, curr_target, adj_mapping, adj_matrix_predicted)
            cate_predicted.append(intervention_prediction - reference_predidction)
        cate_predicted = np.array(cate_predicted)
        all_cate_predicted.append(cate_predicted)
    all_cate_predicted = np.array(all_cate_predicted)
    np.save('cate_estimate_private.npy', all_cate_predicted)    # cate_estimate.npy for private task


if __name__ == '__main__':
    main()