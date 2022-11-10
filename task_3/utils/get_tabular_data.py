#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json, os, random
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from .static import *

with open(f'./data/inoutpair/user_construct_event_sequence_timeindex.json', 'r') as f: user_construct_event_sequence_timeindex = json.load(f)
user_construct_event_sequence_timeindex = {int(user): v for user, v in user_construct_event_sequence_timeindex.items()}
construct_idx_to_users = [set() for _ in range(116)]
# a list in length 116, each element is a set of users. here users names are integers
for user, construct_event_sequence in user_construct_event_sequence_timeindex.items():
    this_users_construct = [int(k) for k in construct_event_sequence.keys() if k != 'max_index']
    this_users_construct_idxs = [constructs_to_idx[cid] for cid in this_users_construct]
    for cix in this_users_construct_idxs: construct_idx_to_users[cix].add(user)
with open(f'./data/inoutpair/user_to_timestamps.json', 'r') as f: user_to_timestamps = json.load(f)
user_to_timestamps = {int(user): v for user, v in user_to_timestamps.items()}

def linear_interpolate_a_table(raw_table, real_observed_stamps):
    '''
    Args:
        raw_table: np array in shape (#time_steps, # constructs)
        real_observed_stamps: dict with key: cid in range [0, #constructs] and value being timestamps list
            note that the keys set is usually a small subset of whole [#constructs] set.
            note that the values of this dict should have no intersection with each other
    Returns: a new table where each column that has real observed values are filled in
    '''
    REAL_OBSV_MASK = np.zeros_like(raw_table).astype(bool)
    real_observed_cids = list(real_observed_stamps.keys())
    unobserved_cids = list(set(range(116)) - set(real_observed_cids))
    for cid, stamps in real_observed_stamps.items(): REAL_OBSV_MASK[stamps, cid] = True
    new_table = np.copy(raw_table)
    new_table[~REAL_OBSV_MASK] = np.nan  # first we draw all the missing values as nan

    for cid in real_observed_cids:
        stamps = real_observed_stamps[cid]
        if len(stamps) == 1:
            new_table[:, cid] = new_table[stamps[0], cid] # to make the whole column the same
            continue
        # now we have more than one observed stamps
        real_values = new_table[stamps, cid]
        interp_f = interpolate.interp1d(stamps, real_values, kind='linear', ) # fill_value='extrapolate'
        minstamp, maxstamp = stamps[0], stamps[-1] # min(stamps), max(stamps)
        rv_tmin, rv_tmax = real_values[0], real_values[-1]
        new_table[minstamp:maxstamp+1, cid] = interp_f(range(minstamp, maxstamp+1))
        new_table[:minstamp, cid], new_table[maxstamp+1:, cid] = rv_tmin, rv_tmax

    assert np.all(new_table[REAL_OBSV_MASK] == raw_table[REAL_OBSV_MASK])
    assert not np.any(np.isnan(new_table[:, real_observed_cids])) # observed cids are filled in for whole sequence
    assert np.all(np.isnan(new_table[:, unobserved_cids])) # unobserved cids are all nan
    return new_table


def EVENT_TO_EXACT_SCORE(event):
    # now we dont care about gain or learning or accumulated knowledge level; just the currrent performance
    # for all possible rvents, see ./all_possible_event_for_a_question_pair_session_and_count.txt
    # im not sure whether this rubric is a fair (or with good performance) one
    rubric = EVENT_TO_EXACT_SCORE_rubric
    learned_a_lesson = 'Lesson' in event
    if event in rubric: return rubric[event], learned_a_lesson
    correct_count, wrong_count = event.count('correct'), event.count('wrong')
    return 100 * correct_count / (correct_count + wrong_count), learned_a_lesson

def EVENT_TO_GAIN_SCORE(event):
    rubric = EVENT_TO_GAIN_SCORE_rubric
    learned_a_lesson = 'Lesson' in event
    if event in rubric: return rubric[event], learned_a_lesson
    lesson_gain = +10 if learned_a_lesson else 0
    correct_count, wrong_count = event.count('correct'), event.count('wrong')
    performance_gain = 10 * correct_count / (correct_count + wrong_count) # not very accurate. but only a small portion of data falls here
    return lesson_gain + performance_gain, learned_a_lesson

def get_exact_scores():
    '''
    Returns: a dictionary
    '''
    final_dict = {}
    for user, construct_event_sequence in user_construct_event_sequence_timeindex.items():
        max_ind = construct_event_sequence['max_index']
        raw_scores_table = np.zeros((max_ind + 1, 116))
        learned_lessons = np.ones((max_ind + 1)) * np.nan # nan means nothing learned
        real_observed_stamps = {}
        for construct, event_sequence in construct_event_sequence.items():
            if construct == 'max_index': continue
            cid = constructs_to_idx[int(construct)]
            real_observed_stamps[cid] = []
            for tstep, event in event_sequence:
                real_observed_stamps[cid].append(tstep)
                raw_scores_table[tstep, cid], learned_flag = EVENT_TO_EXACT_SCORE(event)
                if learned_flag: learned_lessons[tstep] = cid
        final_dict[user] = {
            'scores': linear_interpolate_a_table(raw_scores_table, real_observed_stamps),
            'learned_lessons': learned_lessons,
            'real_observed_stamps': real_observed_stamps,
        }
    return final_dict

def get_accumulated_scores(normed=False):
    '''
    Returns: a dictionary
    '''
    final_dict = {}
    for user, construct_event_sequence in user_construct_event_sequence_timeindex.items():
        max_ind = construct_event_sequence['max_index']
        raw_accu_scores_table = np.zeros((max_ind + 1, 116))
        learned_lessons = np.ones((max_ind + 1)) * np.nan # nan means nothing learned
        real_observed_stamps = {}
        for construct, event_sequence in construct_event_sequence.items():
            if construct == 'max_index': continue
            cid = constructs_to_idx[int(construct)]
            real_observed_stamps[cid] = []
            current_score = 0
            for tstep, event in event_sequence:
                real_observed_stamps[cid].append(tstep)
                gain_score, learned_flag = EVENT_TO_GAIN_SCORE(event)
                current_score += gain_score
                raw_accu_scores_table[tstep, cid] = current_score
                if learned_flag: learned_lessons[tstep] = cid
            if normed:
                min_of_this_column = np.nanmin(raw_accu_scores_table[:, cid])
                max_of_this_column = np.nanmax(raw_accu_scores_table[:, cid])

                # norm into [0, 1] range (of one's knowledge level)
                if max_of_this_column != min_of_this_column:
                    raw_accu_scores_table[:, cid] = (raw_accu_scores_table[:, cid] - min_of_this_column) / \
                                                        (max_of_this_column - min_of_this_column)
                elif max_of_this_column != 0:
                    raw_accu_scores_table[:, cid] = raw_accu_scores_table[:, cid] / max_of_this_column
        observ_stamp_with_0_ahead = {}
        for cid, stamps in real_observed_stamps.items():
            observ_stamp_with_0_ahead[cid] = [0] + stamps if 0 not in stamps else stamps
        final_dict[user] = {
            'scores': linear_interpolate_a_table(raw_accu_scores_table, observ_stamp_with_0_ahead),
            'learned_lessons': learned_lessons,
            'real_observed_stamps': real_observed_stamps,
        }
    return final_dict

