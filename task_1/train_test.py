#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
In this file, we train two types of models:
    1. student model: for each pair (i, j), each student gives a score, and this score is for 3-class classification
    2. an aggregate model: for each pair (i, j), we aggregate the scores from all students, and additional features from 'stacked',
        and predict whether the edge exists or not (binary classification)
"""
import random, json, os, joblib, pickle
import numpy as np
from evaluation.adjacency_utils import edge_prediction_metrics
import lightgbm as lgb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _get_stat(a_list_of_values, percentiles=(5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)):
    if len(a_list_of_values) == 0: a_list_of_values = [0.]
    return np.append([np.min(a_list_of_values), np.max(a_list_of_values),
                      np.mean(a_list_of_values), np.std(a_list_of_values)],
                     np.percentile(a_list_of_values, percentiles))


p_c_head_list = [(p, c) for p in range(50) for c in range(50) if p != c]
reverse_p_c_head_index = {p_c_head: index * 100 for index, p_c_head in enumerate(p_c_head_list)}

def train(train_dataset_ids=None, seed=1, num_bagging=5):
    set_seed(seed)
    if train_dataset_ids is None: train_dataset_ids = [0, 1, 2, 3, 4]
    training_id_to_rank = {id: rank for rank, id in enumerate(train_dataset_ids)}

    def _get_locations(head, fts):
        # fts is to make sure that the result is correct
        did, ci, cj = head
        rk = training_id_to_rank[did]
        pure_locations_start = 245000 * rk + reverse_p_c_head_index[(ci, cj)]
        pure_locations = np.arange(pure_locations_start, pure_locations_start + 100) # 100 students
        assert (fts[pure_locations][:, [0, 1, 2]] == list(head)).all()
        return pure_locations

    num_constructs = 50

    features_single_students_all = [np.load(f'./saved_features/features_local_dev_{did}_each_single_students.npy')
                                    for did in train_dataset_ids]
    labels_single_students_all = [np.load(f'./saved_features/labels_local_dev_{did}_each_single_students.npy')
                                  for did in train_dataset_ids]

    features_train_all = np.vstack([np.hstack([np.ones((len(features_single_students_all[_]), 1)) * did,
                                               features_single_students_all[_]]) for _, did in enumerate(train_dataset_ids)])
    labels_train_all = np.hstack(labels_single_students_all)
    assert labels_train_all.max() == 2

    all_ordered_heads = [(did, ci, cj)
                         for did in train_dataset_ids
                         for ci in range(num_constructs)
                         for cj in range(ci + 1, num_constructs)]

    features_stacked_students_all = [np.load(f'./saved_features/features_local_dev_{did}_stacked_students.npy')
                                     for did in range(5)]
    features_to_train_aggregate_model = []
    labels_to_train_aggregate_model = []

    all_data_bagging_indices = list(range(len(all_ordered_heads)))
    random.shuffle(all_data_bagging_indices)
    all_data_bagging_indices = [all_data_bagging_indices[i::num_bagging] for i in range(num_bagging)]
    for i in range(num_bagging):
        # Randomly split the data into 80% for training the student model, and 20% for training aggretation model
        # Then iterate over different 20% of the data
        agg_bagging_indices = all_data_bagging_indices[i]
        student_bagging_indices = list(set(range(len(all_data_bagging_indices))) - {i})
        student_bagging_indices = [all_data_bagging_indices[indices] for indices in student_bagging_indices]    # List of lists
        student_bagging_indices  = [val for sublist in student_bagging_indices for val in sublist]    # Flatten the list of lists

        ordered_heads_to_train_student_model = [all_ordered_heads[i] for i in student_bagging_indices]
        ordered_heads_to_train_aggregate_model = list(set(all_ordered_heads) - set(ordered_heads_to_train_student_model))
        bidirct_heads_to_train_student_model = ordered_heads_to_train_student_model \
                                                + [(did, cj, ci) for did, ci, cj in ordered_heads_to_train_student_model]
        bidrct_heads_to_train_aggregate_model = ordered_heads_to_train_aggregate_model \
                                                + [(did, cj, ci) for did, ci, cj in ordered_heads_to_train_aggregate_model]
        indexs = set().union(*[_get_locations(head, features_train_all) for head in bidirct_heads_to_train_student_model])
        cond_train_student_model = np.zeros(len(features_train_all), dtype=bool)
        cond_train_student_model[list(indexs)] = True
        features_to_train_student_model = features_train_all[cond_train_student_model][:, 3:]
        labels_to_train_student_model = labels_train_all[cond_train_student_model]

        student_clf = lgb.LGBMClassifier()
        student_clf.fit(features_to_train_student_model, labels_to_train_student_model)
        print('training accuracy on student: {}'.format(student_clf.score(features_to_train_student_model,
                                                                          labels_to_train_student_model)))
        student_scores_all = student_clf.predict_proba(features_train_all[:, 3:])
        print('validation accuracy on student: {}'.format(student_clf.score(features_train_all[~cond_train_student_model][:, 3:],
                                                                            labels_train_all[~cond_train_student_model])))

        for (did, ci, cj) in bidrct_heads_to_train_aggregate_model:
            cond_same = _get_locations((did, ci, cj), features_train_all)
            cond_invs = _get_locations((did, cj, ci), features_train_all)
            true_label = labels_train_all[cond_same][0]
            # guaranteed to be the same student per row
            score_same_invs = np.hstack([student_scores_all[cond_same], student_scores_all[cond_invs]])
            stat = np.apply_along_axis(_get_stat, 0, score_same_invs).T.flatten()
            that_index = np.where((features_stacked_students_all[did][:, 0] == ci)
                                  & (features_stacked_students_all[did][:, 1] == cj))[0][0]
            this_index = np.where((features_stacked_students_all[did][:, 0] == ci)
                                  & (features_stacked_students_all[did][:, 1] == cj))[0][0]
            features_to_train_aggregate_model.append(np.hstack([stat, features_stacked_students_all[did][that_index, 2:],
                                                                features_stacked_students_all[did][this_index, 2:]]))
            labels_to_train_aggregate_model.append(true_label)

        # save model
        svdir = './saved_models'
        os.makedirs(svdir, exist_ok=True)
        joblib.dump(student_clf, os.path.join(svdir, 'student_clf_{}.pkl'.format(i)))

    ######################AGGREGATE MODEL IS BINARY CLASSIFIER  ############################
    labels_to_train_aggregate_model = np.array(labels_to_train_aggregate_model)
    ind2 = labels_to_train_aggregate_model == 2
    ind1 = labels_to_train_aggregate_model == 1
    ind0 = labels_to_train_aggregate_model == 0
    labels_to_train_aggregate_model[ind2] = 0
    labels_to_train_aggregate_model[ind1] = 0
    labels_to_train_aggregate_model[ind0] = 1
    ######################AGGREGATE MODEL IS BINARY CLASSIFIER  ############################

    # print(f'start training aggregate model: {np.array(features_to_train_aggregate_model).shape}')
    aggregate_clf = lgb.LGBMClassifier()
    aggregate_clf.fit(np.vstack(features_to_train_aggregate_model), np.array(labels_to_train_aggregate_model))
    print('training accuracy on aggregation: ', aggregate_clf.score(np.vstack(features_to_train_aggregate_model),
                                                                    np.array(labels_to_train_aggregate_model)))
    # save model
    svdir = './saved_models'
    joblib.dump(aggregate_clf, os.path.join(svdir, 'aggregate_clf.pkl'))


def predict(name, val_dataset_ids=None, num_bagging=5):
    if val_dataset_ids is None: val_dataset_ids = [0, 1, 2, 3, 4]

    svdir = './saved_models'
    student_clfs = [joblib.load(os.path.join(svdir, 'student_clf_{}.pkl'.format(i))) for i in range(num_bagging)]
    aggregate_clf = joblib.load(os.path.join(svdir, 'aggregate_clf.pkl'))
    features_stacked_students_all \
        = [np.load(f'./saved_features/features_{name}_{did}_stacked_students.npy') for did in range(5)]

    if name == 'local_dev':
        dag_true_path = f'./data/Task_1_data_local_dev_csv/adj_matrix.npy'
        adj_matrix_true_stacks = np.load(dag_true_path).astype(bool)

    ALL_RES_STACK = []
    for dataset_id in val_dataset_ids:
        print(f'\n--- validating on dataset {name}-{dataset_id} ---')
        features_test = np.load(f'./saved_features/features_{name}_{dataset_id}_each_single_students.npy')
        # student_scores = student_clf.predict_proba(features_test[:, 2:])
        student_scores = np.array([student_clf.predict_proba(features_test[:, 2:]) for student_clf in student_clfs])
        student_scores = np.swapaxes(student_scores, 1, 0)
        student_scores = student_scores.mean(axis=1)

        cicj_pairs = list({(ci, cj) for ci, cj in features_test[:, :2].astype(int)})
        features_into_aggregate_model = []

        for ci, cj in cicj_pairs:
            cond_same_start = reverse_p_c_head_index[(ci, cj)]
            cond_same = np.arange(cond_same_start, cond_same_start + 100)
            cond_invs_start = reverse_p_c_head_index[(cj, ci)]
            cond_invs = np.arange(cond_invs_start, cond_invs_start + 100)
            assert (features_test[cond_same][:, :2] == [ci, cj]).all()
            assert (features_test[cond_invs][:, :2] == [cj, ci]).all()
            score_same_invs = np.hstack([student_scores[cond_same], student_scores[cond_invs]])
            stat = np.apply_along_axis(_get_stat, 0, score_same_invs).T.flatten()
            that_index = np.where((features_stacked_students_all[dataset_id][:, 0] == ci)
                                  & (features_stacked_students_all[dataset_id][:, 1] == cj))[0][0]
            this_index = np.where((features_stacked_students_all[dataset_id][:, 0] == cj)
                                  & (features_stacked_students_all[dataset_id][:, 1] == ci))[0][0]
            features_into_aggregate_model.append(np.hstack([stat,
                                                            features_stacked_students_all[dataset_id][that_index, 2:],
                                                            features_stacked_students_all[dataset_id][this_index, 2:]]))

        cicj_pairs = np.array(cicj_pairs).astype(int)
        final_scores = aggregate_clf.predict_proba(np.vstack(features_into_aggregate_model))[:, 1]
        scores_dict = {(ci, cj): final_scores[i] for i, (ci, cj) in enumerate(cicj_pairs)}

        adj_matrix_predicted = np.zeros((50, 50)).astype(bool)
        for ci, cj in cicj_pairs:
            if scores_dict[(ci, cj)] > 0.5 and scores_dict[(ci, cj)] > scores_dict[(cj, ci)]:
                adj_matrix_predicted[ci, cj] = True

        ALL_RES_STACK.append(adj_matrix_predicted)

        if name == 'local_dev':
            finalres = edge_prediction_metrics(adj_matrix_true_stacks[dataset_id], adj_matrix_predicted)
            print(f'  final result:', {'precision': finalres['orientation_precision'],
                                       'recall': finalres['orientation_recall'],
                                       'fscore': finalres['orientation_fscore'],
                                       'numedges': int(adj_matrix_predicted.sum())}, '\n')

    np.save(os.path.join(svdir, f'adj_matrix_{name}.npy'), np.array(ALL_RES_STACK))


if __name__ == '__main__':
    num_bagging = 5
    seed = 1
    train([_ for _ in range(5)], seed, num_bagging)
    predict('private', [_ for _ in range(5)], num_bagging)