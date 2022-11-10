#!/usr/bin/python3
# -*- coding: utf-8 -*-
from itertools import chain, combinations
from collections import Counter
from scipy.spatial import distance
import json, os, random
import matplotlib.pyplot as plt
import numpy as np

from utils.static import *
from utils.get_tabular_data import constructs, constructs_to_idx, \
    user_construct_event_sequence_timeindex, construct_idx_to_users, \
    get_exact_scores, get_accumulated_scores
from utils.constructs_more_info import c_more_info_dict
with open(f'./data/inoutpair/user_construct_event_sequence.json', 'r') as f: user_construct_event_sequence = json.load(f)

######################## pairwise handcrafted heuristics ########################
'''
with the hypothesis that ci->cj, we would expect:
1. after the user [learns] ci, his performance on cj will improve (comparing to times before learning)
2. after the user [reaches] some level on ci, his performance on cj will improve
3. comparing to users who have never did ci, users who did ci (in lifetime) will have higher performance on cj
4. consider together with e.g., timelagdiffs, ...
'''
SCORES = {
    # each returns a dict with keys: 'scores', 'learned_lessons', and 'real_observed_stamps'
    'exact_scores': get_exact_scores(),
    'accumulated_scores_unnormed': get_accumulated_scores(normed=False),
    'accumulated_scores_normed': get_accumulated_scores(normed=True),
}

def get_co_occurence_matrices(allow_repeat=True):
    co_occurance_matrix = np.zeros((len(constructs), len(constructs)))
    this_construct_to_users = [[] for _ in range(len(constructs))] # already renamed; users may be repeated

    for user, construct_event_sequence in user_construct_event_sequence_timeindex.items():
        cons_counts = {int(cid): len(event_sequence) for cid, event_sequence in
                       construct_event_sequence.items() if cid != 'max_index'}
        cons_counts_list = [constructs_to_idx[cid] for cid, count in cons_counts.items() for _ in range(count)]
        cons = cons_counts_list if allow_repeat else list(set(cons_counts_list))
        for cid in cons: this_construct_to_users[cid].append(user) # users may be repeated
        cons_pairs = list(combinations(cons, 2))
        for coni, conj in cons_pairs:
            co_occurance_matrix[coni, conj] += 1
            co_occurance_matrix[conj, coni] += 1
    for i in range(len(constructs)): co_occurance_matrix[i, i] /= 2

    # normalize. not directly, but over "people who did it"
    # m[i, j] \in [0, 1]: among all people who did i, how many of them also did j. i.e., p(did i&j | did i)
    num_of_users_on_each_construct = np.array(list(map(len, this_construct_to_users))) # i.e., the self-frequency of each construct
    if not allow_repeat: assert np.all(co_occurance_matrix.max(axis=1) <= num_of_users_on_each_construct)
    normed_co_occurance_matrix = co_occurance_matrix / num_of_users_on_each_construct[:, None]
    normed_co_occurance_matrix_str = [
        [f'{int(x)}/{num_of_users_on_each_construct[rid]}' for x in row] \
            for rid, row in enumerate(co_occurance_matrix)
    ]

    # also normalize, but symmetric by venn graph.
    # m[i, j]=m[j, i] \in [0, 1]: among all people who did i or j, how many of them did both i and j. i.e., p(did AND | did OR)
    venn_co_occurance_matrix = np.zeros((len(constructs), len(constructs)))
    venn_co_occurance_matrix_str = [[None for ii in range(len(constructs))] for jj in range(len(constructs))]
    for i in range(len(constructs)):
        for j in range(i+1, len(constructs)):
            usersi = this_construct_to_users[i]
            usersj = this_construct_to_users[j]
            num_users_iandj = len(list((Counter(usersi) & Counter(usersj)).elements())) # len(usersi & usersj)
            num_users_iorj = len(list((Counter(usersi) | Counter(usersj)).elements())) # len(usersi | usersj)
            ratio = num_users_iandj / num_users_iorj
            venn_co_occurance_matrix[i, j] = venn_co_occurance_matrix[j, i] = ratio
            venn_co_occurance_matrix_str[i][j] = venn_co_occurance_matrix_str[j][i] = f'{num_users_iandj}/{num_users_iorj}'

    jaccard_co_occurance_matrix = 1 - distance.cdist(co_occurance_matrix, co_occurance_matrix, 'jaccard')

    MATS = {
        'co_occurance_matrix': co_occurance_matrix,
        'normed_co_occurance_matrix': normed_co_occurance_matrix,
        'venn_co_occurance_matrix': venn_co_occurance_matrix,
        'jaccard_co_occurance_matrix': jaccard_co_occurance_matrix,
    }

    # here *co_occurance_matrix_str helps you to understand how each entry is calculated (especially when denominator is small)
    # you may also plot the co-occurance matrices as a heatmap and see the thresholds/patterns,
    # you may also print the pairs where they have largest/smallest co-occurance, and find that they are reasonable (ask haoyue for it).
    return MATS

def get_whole_scatter_correlation(ci, cj, scoretype='accumulated_scores_normed'):
    # following this idea, we can find the correlation among this bivariate
    # note that here ci, cj are in range(len(constructs)). same for all functions in this file.
    svdir = f'./pairwise_plots/scatters/{scoretype}'
    os.makedirs(svdir, exist_ok=True)
    users_did_both_ij = construct_idx_to_users[ci] & construct_idx_to_users[cj]
    if len(users_did_both_ij) == 0: return np.nan
    ci_scores = np.hstack([SCORES[scoretype][user]['scores'][:, ci] for user in users_did_both_ij])
    cj_scores = np.hstack([SCORES[scoretype][user]['scores'][:, cj] for user in users_did_both_ij])
    return np.corrcoef(ci_scores, cj_scores)[0, 1]

def get_ci_cj_timelag(ci, cj):
    '''
    imagine: in a user's history, if every time ci occurs earlier than cj, i.e., in the quizzes assigned to them,
        ci is always earlier, then we may guess that the platform already thinks that ci->cj.
        Moreover, if the timelag is always very close,
         user does ci at timesteps [100, 200, 300, ...]
         user does cj at timesteps [98, 195, 290, ...],
        we can be more confident about their correlation (and direction in platform's mind).
    so, if the timelag is always negative with a small abs, small variance -> we can be more confident about the direction.
    '''
    users_did_both_ij = construct_idx_to_users[ci] & construct_idx_to_users[cj]
    if len(users_did_both_ij) == 0: return np.nan

    # for both lists, we expect small negative values with small variance, to indicate ci->cj.
    time_lag_diffs_i_to_j = [] # for each ci timestep, find the nearest cj timestep, and record cit-cjt.
    time_lag_diffs_j_to_i = [] # for each cj timestep, find the nearest ci timestep, and record cit-cjt.
    for user in users_did_both_ij:
        num_whole_history = len(SCORES['exact_scores'][user]['scores'])
        ci_timesteps = SCORES['exact_scores'][user]['real_observed_stamps'][ci] # which scoretype doesn't matter
        cj_timesteps = SCORES['exact_scores'][user]['real_observed_stamps'][cj]
        for cit in ci_timesteps:
            closest_cjt = min(cj_timesteps, key=lambda x: abs(x-cit))
            time_lag_diffs_i_to_j.append((cit - closest_cjt) / num_whole_history) # normalize by the whole history length in in [0, 1]
        for cjt in cj_timesteps:
            closest_cit = min(ci_timesteps, key=lambda x: abs(x-cjt))
            time_lag_diffs_j_to_i.append((closest_cit - cjt) / num_whole_history)
    return np.mean(time_lag_diffs_i_to_j + time_lag_diffs_j_to_i)
           # np.std(time_lag_diffs_i_to_j + time_lag_diffs_j_to_i), \
           # np.median(time_lag_diffs_i_to_j + time_lag_diffs_j_to_i)

def get_ci_cj_accuracy_diff(ci, cj):

    users_did_both_ij = construct_idx_to_users[ci] & construct_idx_to_users[cj]
    if len(users_did_both_ij) == 0: return np.nan, np.nan, np.nan, np.nan, np.nan

    ##########################################################
    MAX_AFFECT_RATIO = 0.2
    cj_scores_before_after_learn_ci = {'before': [], 'after': []}
    for user in users_did_both_ij:
        learned_ci_inds = np.where(SCORES['exact_scores'][user]['learned_lessons'] == ci)[0]
        if len(learned_ci_inds) == 0: continue  # no ci learning in history
        middle_learn_index = (len(learned_ci_inds) - 1) // 2 # we just pick a middle learning timestep
        learned_ci_ind = learned_ci_inds[middle_learn_index]

        cj_scores = SCORES['accumulated_scores_normed'][user]['scores'][:, cj]  # we want to use exact scores instead of accumulated (which must be diffed)
        cj_scores_before, cj_scores_after = \
            cj_scores[learned_ci_ind] - cj_scores[:learned_ci_ind], \
            cj_scores[learned_ci_ind:] - cj_scores[learned_ci_ind]  # diff to current level (see change speed)

        MAX_AFFECT_RANGE = int(MAX_AFFECT_RATIO * len(cj_scores))  # we only consider the nearest of the history before/after learning
        cj_scores_before, cj_scores_after = cj_scores_before[-MAX_AFFECT_RANGE:], cj_scores_after[:MAX_AFFECT_RANGE]
        cj_scores_before_after_learn_ci['before'].extend(cj_scores_before)
        cj_scores_before_after_learn_ci['after'].extend(cj_scores_after)

    ##########################################################
    MAX_AFFECT_RATIO = 0.2
    reachT = 0.5
    cj_scores_before_after_ci_reach_T = {'before': [], 'after': []}
    for user in users_did_both_ij:
        ci_scores = SCORES['accumulated_scores_normed'][user]['scores'][:, ci]
        the_first_reach_T_inds = np.where(ci_scores >= reachT)[0]
        if len(the_first_reach_T_inds) == 0: continue  # no ci reaching T in history

        cj_scores = SCORES['accumulated_scores_normed'][user]['scores'][:, cj]
        cj_scores_before, cj_scores_after = \
            cj_scores[the_first_reach_T_inds[0]] - cj_scores[:the_first_reach_T_inds[0]], \
            cj_scores[the_first_reach_T_inds[0]:] - cj_scores[the_first_reach_T_inds[0]]  # diff to current level (see change speed)
        MAX_AFFECT_RANGE = int(MAX_AFFECT_RATIO * len(cj_scores))  # we only consider the nearest of the history before/after learning
        cj_scores_before, cj_scores_after = cj_scores_before[-MAX_AFFECT_RANGE:], cj_scores_after[:MAX_AFFECT_RANGE]
        cj_scores_before_after_ci_reach_T['before'].extend(cj_scores_before)
        cj_scores_before_after_ci_reach_T['after'].extend(cj_scores_after)

    ##########################################################
    cj_accu_scores_with_without_ci_in_history = {'with': [], 'without': []}
    cj_exct_scores_with_without_ci_in_history = {'with': [], 'without': []}
    users_who_only_did_cj = construct_idx_to_users[cj] - users_did_both_ij
    for user in users_did_both_ij:
        cj_accu_scores_with_without_ci_in_history['with'].extend(SCORES['accumulated_scores_normed'][user]['scores'][:, cj])
        cj_exct_scores_with_without_ci_in_history['with'].extend(SCORES['exact_scores'][user]['scores'][:, cj])
    for user in users_who_only_did_cj:
        cj_accu_scores_with_without_ci_in_history['without'].extend(SCORES['accumulated_scores_normed'][user]['scores'][:, cj])
        cj_exct_scores_with_without_ci_in_history['without'].extend(SCORES['exact_scores'][user]['scores'][:, cj])

    ##########################################################
    reachT = 0.1
    reachT_time_diffs = []
    for user in users_did_both_ij:
        num_history = len(SCORES['accumulated_scores_normed'][user]['scores'])
        ci_scores = SCORES['accumulated_scores_normed'][user]['scores'][:, ci]
        cj_scores = SCORES['accumulated_scores_normed'][user]['scores'][:, cj]
        ci_the_first_reach_T_inds = np.where(ci_scores >= reachT)[0]
        cj_the_first_reach_T_inds = np.where(cj_scores >= reachT)[0]
        if len(ci_the_first_reach_T_inds) == 0 or len(cj_the_first_reach_T_inds) == 0: continue
        diff = (cj_the_first_reach_T_inds[0] - ci_the_first_reach_T_inds[0]) / num_history
        reachT_time_diffs.append(diff)

    return np.nanmean(cj_scores_before_after_learn_ci['after']) - np.nanmean(cj_scores_before_after_learn_ci['before']), \
           np.nanmean(cj_scores_before_after_ci_reach_T['after']) - np.nanmean(cj_scores_before_after_ci_reach_T['before']), \
           np.nanmean(cj_accu_scores_with_without_ci_in_history['with']) - np.nanmean(cj_accu_scores_with_without_ci_in_history['without']), \
           np.nanmean(cj_exct_scores_with_without_ci_in_history['with']) - np.nanmean(cj_exct_scores_with_without_ci_in_history['without']), \
           np.nanmean(reachT_time_diffs)

MATS = get_co_occurence_matrices(allow_repeat=True)
venn_co_occurance_matrix = MATS['venn_co_occurance_matrix'] # symmetric. larger better
co_occurance_matrix = MATS['co_occurance_matrix']
normed_co_occurance_matrix = MATS['normed_co_occurance_matrix']
jaccard_co_occurance_matrix = MATS['jaccard_co_occurance_matrix']

accu_score_correlations = np.zeros((116, 116)) # symmetric. larger better
for i in range(116):
    for j in range(i + 1, 116):
        accu_score_correlations[i, j] = accu_score_correlations[j, i] = \
            get_whole_scatter_correlation(i, j, 'accumulated_scores_normed')

timelags = np.zeros((116, 116)) # not symmetric. only want the negative entries. then under negative (or less than some -T), larger better
for i in range(116):
    for j in range(i + 1, 116):
        ci_than_cj = get_ci_cj_timelag(i, j)
        timelags[i, j] = ci_than_cj
        timelags[j, i] = -ci_than_cj

cj_scores_before_after_learn_ci = np.zeros((116, 116))            # asymmetric. larger better
cj_scores_before_after_ci_reach_T = np.zeros((116, 116))          # asymmetric. larger better
cj_accu_scores_with_without_ci_in_history = np.zeros((116, 116))  # asymmetric. larger better
cj_exct_scores_with_without_ci_in_history = np.zeros((116, 116))  # asymmetric. larger better
cj_reachT_time_diffs_than_ci = np.zeros((116, 116))               # asymmetric. only want positive entries. then under positive (or larger than some T), smaller better

for i in range(116):
    for j in range(i + 1, 116):
        cj_beforeafter_ci_learn, cj_beforeafter_ci_reach_T, cj_accu_with_without_ci, cj_exct_with_without_ci, cj_reachT_time_diff = \
            get_ci_cj_accuracy_diff(i, j)
        cj_scores_before_after_learn_ci[i, j] = cj_beforeafter_ci_learn
        cj_scores_before_after_ci_reach_T[i, j] = cj_beforeafter_ci_reach_T
        cj_accu_scores_with_without_ci_in_history[i, j] = cj_accu_with_without_ci
        cj_exct_scores_with_without_ci_in_history[i, j] = cj_exct_with_without_ci
        cj_reachT_time_diffs_than_ci[i, j] = cj_reachT_time_diff
######################## pairwise handcrafted heuristics ########################

######################## BKT params heuristics ########################
params_dict = {cid: {} for cid in range(116)}
for collect_type in ['eachstamp', 'inoutpair']:
    bkt_params = pd.read_csv(f'./data/{collect_type}/bkt/params.csv')
    for _, row in bkt_params.iterrows():
        params_dict[row['skill']][f"{collect_type}_{row['param']}"] = row['value']
######################## BKT params heuristics ########################

######################## meta singleton heuristics ########################
# just some seemingly useful examples; according to c_more_info_dict we can go more
SINGLETON_STATS = {}
PMIN, PMAX = 10, 90
for cid in constructs:
    this_cid_stats = {}
    user_to_age, user_to_year_group, user_to_corrects = {}, {}, {}
    for dd in c_more_info_dict[cid]['quiz_sessions_info_containing_this_in_data']:
        uid, age, ygrp = dd['user_id'], parse_time(dd['age_now']), dd['user_year_group']
        user_to_age[uid] = age  # with refresh
        user_to_year_group[uid] = ygrp
        if uid not in user_to_corrects: user_to_corrects[uid] = [dd['iscorrect']]
        else: user_to_corrects[uid].append(dd['iscorrect'])

    ages, year_groups = list(user_to_age.values()), list(user_to_year_group.values())
    this_cid_stats['users_year_group_mean'] = np.nanmean(year_groups)

    accuracies, corrects, whole_corrects = [], [], []
    for corrects_list in user_to_corrects.values():
        corrects_list = [_ for _ in corrects_list if _ is not None]
        corrects.append(np.sum(corrects_list))
        accuracies.append(np.mean(corrects_list))
        whole_corrects.extend(corrects_list)
    this_cid_stats['accuracies_total'] = np.nanmean(whole_corrects)  # marginal accuracy among students

    meta_levels = [dd['level'] for dd in c_more_info_dict[cid]['quiz_subject_info_containing_this_in_meta']]
    meta_quiz_sequences = [dd['quiz_sequence'] for dd in c_more_info_dict[cid]['quiz_subject_info_containing_this_in_meta']]
    meta_question_sequences = [dd['question_sequence'] for dd in c_more_info_dict[cid]['quiz_subject_info_containing_this_in_meta']]

    if len(meta_levels) == 0: meta_levels = [np.nan]
    if len(meta_quiz_sequences) == 0: meta_quiz_sequences = [np.nan]
    if len(meta_question_sequences) == 0: meta_question_sequences = [np.nan]
    this_cid_stats['meta_quiz_sequences_max'] = np.nanpercentile(meta_quiz_sequences, PMAX)
    SINGLETON_STATS[constructs_to_idx[cid]] = this_cid_stats
######################## meta singleton heuristics ########################

######################## pairwise imputed data heuristics ########################
original_imputed_dict = {
    'inoutpair_hand_exact_scores': np.load('./data/inoutpair/handcraft/exact_scores.npy'),
    'inoutpair_hand_accumulated_scores': np.load('./data/inoutpair/handcraft/accumulated_scores_normed.npy'),
    'inoutpair_bkt': np.load('./data/inoutpair/bkt/preds.npy'),
    'inoutpair_dkt': np.load('./data/inoutpair/dkt/preds.npy'),
    'eachstamp_bkt': np.load('./data/eachstamp/bkt/preds.npy'),
    'eachstamp_dkt': np.load('./data/eachstamp/dkt/preds.npy'),
}
big_array_indexs_cache = {}
for imp_name, scores in original_imputed_dict.items():
    big_array_indexs_cache[imp_name] = {}
    for usr in np.unique(scores[:, 0]):
        big_array_indexs_cache[imp_name][usr] = np.where(scores[:, 0] == usr)[0]
#
PAIRWISE_WAVELETS_STATS = {}
for imp_name, scores in original_imputed_dict.items():
    stats = {_: np.zeros((116, 116)) for _ in
             ['correlations_mean', 'whole_correlation', 'whole_step_differences_mean']}
    for i in range(116):
        print(f'imp_name: {imp_name}, i: {i}')
        for j in range(116):
            if i == j: continue
            users_i = construct_idx_to_users[i]
            users_j = construct_idx_to_users[j]
            users = list(set(users_i) & set(users_j))

            i_sequences, j_sequences = [], []
            for usr in users:
                idxs = big_array_indexs_cache[imp_name][usr]
                i_sequences.append(scores[idxs, i + 1])
                j_sequences.append(scores[idxs, j + 1])
            correlations = [np.corrcoef(i_seq, j_seq)[0, 1] for i_seq, j_seq in zip(i_sequences, j_sequences)]
            stats['correlations_mean'][i, j] = np.nanmean(correlations)
            i_sequences, j_sequences = np.concatenate(i_sequences), np.concatenate(j_sequences)
            stats['whole_correlation'][i, j] = np.corrcoef(i_sequences, j_sequences)[0, 1]
            whole_step_differences = i_sequences - j_sequences
            stats['whole_step_differences_mean'][i, j] = np.nanmean(whole_step_differences)
    PAIRWISE_WAVELETS_STATS[imp_name] = stats
PAIRWISE_WAVELETS_STATS = {f'{imp_name}_{stat_name}': statMAT for imp_name, stats in PAIRWISE_WAVELETS_STATS.items() for stat_name, statMAT in stats.items()}
######################## pairwise imputed data heuristics ########################

######################## pairwise fake do() operations change heuristics ########################
DO_WHOLE_DICT = {}
for imp_name in ['inoutpair_dkt', 'eachstamp_dkt']:
    original_scores = original_imputed_dict[imp_name]
    for i in range(116):
        print(f'processing {imp_name} {i}')
        for j in range(116):
            if i == j: continue
            users_i = construct_idx_to_users[i]
            users_j = construct_idx_to_users[j]
            users_both = list(set(users_i) & set(users_j))
            users_both_indexs = [big_array_indexs_cache[imp_name][usr] for usr in users_both]

            four_sequences = {'original': original_scores[users_both_indexs, j+1]}
            for experiment_name in ['removed', 'allwin', 'alllose']:
                fake_score = np.load(f'./data/{imp_name.split("_")[0]}/dkt/preds_{experiment_name}_{i}.npy')
                if fake_score.shape == original_scores.shape: four_sequences[experiment_name] = fake_score[users_both_indexs, j+1]
                else: four_sequences[experiment_name] = original_scores[users_both_indexs, j+1]

            for exp1, exp2 in [('removed', 'allwin'), ('removed', 'alllose')]:
                # we specifically care about do(allwin) and do(alllose) relative to do(removed) (i.e., no access to ci)
                seq1, seq2 = four_sequences[exp1], four_sequences[exp2]
                significant_indexs = np.where(np.abs(seq1 - seq2) > 0.01)[0]
                significant_ratio = significant_indexs.shape[0] / seq1.shape[0]
                diffs = seq1[significant_indexs] - seq2[significant_indexs]
                diffs_mean = np.nanmean(diffs)
                diffs_min = np.nanpercentile(diffs, PMIN)
                diffs_max = np.nanpercentile(diffs, PMAX)

                for metric_name, mvalue in zip(['significant_ratio', 'diffs_mean', 'diffs_min', 'diffs_max'],
                                              [significant_ratio, diffs_mean, diffs_min, diffs_max]):
                    uniq_key = f'{imp_name}_{exp1}_{exp2}_{metric_name}'
                    if uniq_key not in DO_WHOLE_DICT: DO_WHOLE_DICT[uniq_key] = np.zeros((116, 116))
                    DO_WHOLE_DICT[uniq_key][i, j] = mvalue
######################## pairwise fake do() operations change heuristics ########################

######################## the final step, adjmat prediction (ensemble heuristics above) ########################
######################## require massive parameter finetuning (to the leaderboard) + k-means ##################
useful_heuristics = [
    "cj_scores_before_after_ci_reach_T",
    "cj_scores_before_after_learn_ci",

    "users_year_group_mean_ij_diff",
    "accuracies_total_ij_diff",
    "meta_quiz_sequences_max_ij_diff",

    "eachstamp_dkt_whole_step_differences_mean_ij",
    "inoutpair_bkt_whole_step_differences_mean_ij",

    "eachstamp_dkt_removed_alllose_significant_ratio_ij",
    "inoutpair_dkt_removed_allwin_diffs_min_ij",
    "eachstamp_dkt_removed_alllose_diffs_max_ij",
]
named_masks = {nm: np.zeros((116, 116), dtype=bool) for nm in useful_heuristics}
for i in range(116):
    for j in range(116):
        if i == j: continue

        named_masks['cj_scores_before_after_ci_reach_T'][i, j] = \
            cj_scores_before_after_ci_reach_T[i, j] > -0.01
        named_masks['cj_scores_before_after_learn_ci'][i, j] = \
            cj_scores_before_after_learn_ci[i, j] > -0.01 # in a same sense like ti_tj_timelags

        named_masks['users_year_group_mean_ij_diff'][i, j] = \
            SINGLETON_STATS[i]['users_year_group_mean'] - SINGLETON_STATS[j]['users_year_group_mean'] < 0.5 # causes should be easier (younger)
        named_masks['accuracies_total_ij_diff'][i, j] = \
            SINGLETON_STATS[i]['accuracies_total'] - SINGLETON_STATS[j]['accuracies_total'] > 0. # causes should be easier (higher accuracy)
        named_masks['meta_quiz_sequences_max_ij_diff'][i, j] = \
            SINGLETON_STATS[i]['meta_quiz_sequences_max'] - SINGLETON_STATS[j]['meta_quiz_sequences_max'] < 0. # causes should be easier (lower sequence)

        named_masks['eachstamp_dkt_whole_step_differences_mean_ij'][i, j] = \
            PAIRWISE_WAVELETS_STATS[f'eachstamp_dkt_whole_step_differences_mean'][i, j] > 0.1 # causes should be easier (higher hidden scores learned)
        named_masks['inoutpair_bkt_whole_step_differences_mean_ij'][i, j] = \
            PAIRWISE_WAVELETS_STATS[f'inoutpair_bkt_whole_step_differences_mean'][i, j] > 0.1

        named_masks['eachstamp_dkt_removed_alllose_significant_ratio_ij'][i, j] = \
            DO_WHOLE_DICT['eachstamp_dkt_removed_alllose_significant_ratio'][i, j] > 0.15  # large significant ratio (ci matters a lot for cj)
        named_masks['inoutpair_dkt_removed_allwin_diffs_min_ij'][i, j] = \
            DO_WHOLE_DICT['inoutpair_dkt_removed_allwin_diffs_min'][i, j] < -0.2  # i.e., allwin is a lot larger than removed
        named_masks['eachstamp_dkt_removed_alllose_diffs_max_ij'][i, j] = \
            DO_WHOLE_DICT['eachstamp_dkt_removed_alllose_diffs_max'][i, j] > 0.2  # i.e., alllose is a lot smaller than removed

stacked_masks = np.stack([named_masks[nm] for nm in useful_heuristics], axis=0)
stacked_mean_mask = np.mean(stacked_masks, axis=0)
T_mean_mask = 0.4  # by constraint on the sparsity of the final adjmat. quite strict here
stacked_mean_mask[stacked_mean_mask < T_mean_mask] = 0.
for i in range(116):
    for j in range(i + 1, 116):
        v_ij, v_ji = stacked_mean_mask[i, j], stacked_mean_mask[j, i]
        if v_ij > 0. and v_ji > 0.:
            if v_ij > v_ji: stacked_mean_mask[j, i] = 0.
            else: stacked_mean_mask[i, j] = 0.  # not strict DAG constraint. but just no bi-directional edges

# above is soft mask (by mean ratio); below is a hard mask:
# finally we add an additional co occurence constraint. this is because the ground truth edges are obtained by controled experiments
# then we believe that for any experiment pair (ci, cj), the two must not be too far away from each other.
# e.g., ci is BIDMAS and cj is partial differential equation, one can never conduct experiments for students who didn't learn ci on cj performance
# one possible metric for "distance" would be co occurance (in platform design, the two constructs are always distributed together)
T_venn = 7
additional_mask = venn_co_occurance_matrix > np.percentile(venn_co_occurance_matrix, T_venn)
adjmat = (stacked_mean_mask * additional_mask).astype(bool)


