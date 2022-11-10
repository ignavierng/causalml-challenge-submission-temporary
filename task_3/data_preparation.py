#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from numpyencoder import NumpyEncoder
from utils.static import *

original_csv_path = "../../../../data/Task_3_dataset/checkins_lessons_checkouts_training.csv"
topic_path = '../../../../data/Task_3_dataset/topic_pathway_metadata.csv'
df = pd.read_csv(topic_path)
checkin_to_checkout = {}  # we already know that they are one-to-one mapping
for _, row in df.iterrows(): checkin_to_checkout[row['CheckinQuestionId']] = row['CheckoutQuestionId']
question_id_to_pair_id = {None: -1}  # here pair_id has no real meaning, just a unique id for each pair of in-out questions
for pid, (in_id, out_id) in enumerate(checkin_to_checkout.items()):
    question_id_to_pair_id[in_id] = pid
    question_id_to_pair_id[out_id] = pid

def load_original_data():
    loaded_data = (
        pd.read_csv(original_csv_path, index_col=False)
            .sort_values(["UserId", "Timestamp"], axis=0)  # sort by user id and then by timestamp, not by session ids
            .reset_index(drop=True)
    )
    print('raw loaded, #students:', loaded_data["UserId"].nunique(), '#rows:', len(loaded_data))

    # remove students who accessed very few constructs of interest
    uids, idx, counts = np.unique(np.array(loaded_data["UserId"]), return_counts=True, return_index=True)
    user_ids_accessed_constructs = [loaded_data[idx[c]: idx[c] + counts[c]]["ConstructId"].unique() for c in range(len(uids))]
    user_ids_num_of_constructs_in_input = np.array(
        [len(set(cs).intersection(constructs)) for cs in user_ids_accessed_constructs])
    filter_out = user_ids_num_of_constructs_in_input < 2  # at least we need two variables to show the interaction
    list_idx_to_remove_nest = [list(range(ind, ind + counts[filter_out][c])) for c, ind in enumerate(idx[filter_out])]
    list_idx_to_remove = [v for sublist in list_idx_to_remove_nest for v in sublist]  # flatten it
    loaded_data = loaded_data.drop(labels=list_idx_to_remove, axis=0).reset_index(drop=True)
    print('after remove #constructs<2, #students:', loaded_data["UserId"].nunique(), '#rows:', len(loaded_data))

    # remove rows where the construct is not in the 116 of interests (may be dangerous due to confounders in the rest. but we do it now)
    loaded_data = loaded_data[loaded_data["ConstructId"].isin(constructs)].reset_index(drop=True)
    print('then after remove #constructs not in 116, #students:', loaded_data["UserId"].nunique(),
          '#rows:', loaded_data.shape[0],
          'average #steps per student:', loaded_data.shape[0] / loaded_data["UserId"].nunique())

    # after parsing, we then remove the time sequence length<10 users
    REMOVE_SHORT_THAN = 10
    _, idx, counts = np.unique(np.array(loaded_data["UserId"]), return_counts=True, return_index=True)
    list_idx_to_remove_nest = [list(range(ind, ind + counts[counts <= REMOVE_SHORT_THAN][c])) for c, ind in
                               enumerate(idx[counts <= REMOVE_SHORT_THAN])]
    list_idx_to_remove = [v for sublist in list_idx_to_remove_nest for v in sublist]  # flatten it
    loaded_data = loaded_data.drop(labels=list_idx_to_remove, axis=0).reset_index(drop=True)
    print(f'then after remove #steps<={REMOVE_SHORT_THAN}, #students:', loaded_data["UserId"].nunique(), '#rows:', len(loaded_data))

    # write to local file
    # loaded_data.to_csv('./checkins_lessons_checkouts_training_sorted_filtered.csv', index=False)
    return loaded_data

def parse_data_per_in_out_session():
    # in a sense that the questions are always presented in in-out or in-lesson-out pairs,
    # and the time interval inside each pair/question session should be smaller than the time interval between pairs
    # if we directly separate each timestamp, there might be bias for time intervals.
    # so way compact the in-out sessions for each question (pair) as a single timestamp.
    os.makedirs('./data/inoutpair', exist_ok=True)
    proc_data_user_group = [df for _, df in final_loaded_data.groupby("UserId", as_index=False)]
    whole_dict = {}
    all_time_ranges = []
    user_construct_event_sequence = {}
    for user_data in proc_data_user_group:
        cur_user = user_data["UserId"].iloc[0]
        user_construct_event_sequence[cur_user] = {cid: [] for cid in constructs}

        last_session_id = None
        last_question_sequence = None
        last_construct = None
        scanned_question_sequence = []  # we see questions in-out in a same pair as a same question (e.g., in, lesson, out)
        scanned_question_timestamps = []
        record_raws = []
        visited_sessions = set()

        for row_id_inside_user in range(len(user_data) + 1):  # + 1 for the END
            if row_id_inside_user < len(user_data):
                row = user_data.iloc[row_id_inside_user]
                this_construct = row["ConstructId"]
                assert this_construct in constructs
                this_session_id = row["QuizSessionId"]
                this_question_sequence = row["QuestionSequence"]
                this_question_type = row["Type"]
                still_in_a_same_question = False
                if this_session_id == last_session_id:  # at least should be in a same session
                    if this_question_sequence == last_question_sequence:
                        still_in_a_same_question = True
            else:
                still_in_a_same_question = False  # force to end the last question

            if not still_in_a_same_question:
                whole_str = ', '.join(scanned_question_sequence)
                if whole_str != '':
                    if whole_str not in whole_dict:
                        whole_dict[whole_str] = 1
                    else:
                        whole_dict[whole_str] += 1
                    start_time = datetime.strptime(scanned_question_timestamps[0], "%Y-%m-%d %H:%M:%S.%f")
                    end_time = datetime.strptime(scanned_question_timestamps[-1], "%Y-%m-%d %H:%M:%S.%f")
                    time_range = end_time.timestamp() - start_time.timestamp()
                    all_time_ranges.append(time_range)
                    user_construct_event_sequence[cur_user][last_construct].append((end_time, whole_str))

                scanned_question_sequence = []
                scanned_question_timestamps = []
                record_raws = []

            if this_session_id != last_session_id and this_session_id in visited_sessions: continue  # skip the resumed session
            visited_sessions.add(this_session_id)

            if row_id_inside_user < len(user_data):
                if row["IsCorrect"] == 1: tag = " correct"
                elif row["IsCorrect"] == 0: tag = " wrong"
                else: tag = ""
                scanned_question_sequence.append(f'{row["Type"]}{tag}')
                scanned_question_timestamps.append(row["Timestamp"])
                record_raws.append(row)
                last_session_id = this_session_id
                last_question_sequence = this_question_sequence
                last_construct = this_construct
            else:
                assert len(scanned_question_sequence) == len(scanned_question_timestamps) == len(
                    record_raws) == 0  # i.e., already clear

    # sort whole_dict by values
    whole_dict = {k: v for k, v in sorted(whole_dict.items(), key=lambda item: item[1], reverse=True)}
    strr = ''''''
    for k, v in whole_dict.items(): strr += f'{k} {v}\n'
    with open('./data/inoutpair/all_possible_in_out_session_and_count.txt', 'w') as f: f.write(strr)

    # what is the time range covered by a "question pair"? we suppose it shouldn't be too long (but actually can be days or even months)
    # print(np.mean(all_time_ranges), np.max(all_time_ranges), np.min(all_time_ranges))

    strr = ''''''
    for user, construct_event_sequence in user_construct_event_sequence.items():
        strr += f'===== user {user} =====\n'
        this_user_accessed_constructs = [cid for cid in construct_event_sequence if
                                         len(construct_event_sequence[cid]) > 0]
        assert len(this_user_accessed_constructs) >= 2
        for construct in this_user_accessed_constructs:
            event_sequence = construct_event_sequence[construct]
            strr += f' -- construct {construct} --\n'
            for end_time, event_str in event_sequence:
                strr += f'   {end_time.strftime("%Y-%m-%d %H:%M:%S")} {event_str}\n'
        strr += '\n'

    with open(f'./data/inoutpair/user_construct_event_sequence.txt', 'w') as f:
        f.write(strr)

    clean_user_construct_event_sequence = {
        int(user): {int(cid):
                        [(end_time.strftime("%Y-%m-%d %H:%M:%S"), event_str) for end_time, event_str in event_sequence]
                    for cid, event_sequence in construct_event_sequence.items() if len(event_sequence) > 0}
        for user, construct_event_sequence in user_construct_event_sequence.items()
    }

    with open(f'./data/inoutpair/user_construct_event_sequence.json', 'w') as f:
        json.dump(clean_user_construct_event_sequence, f, indent=4, cls=NumpyEncoder)

    ### then do some timestamps reindex (which of course will lose information on exact time spent, time intervals, etc.)
    user_construct_event_sequence_timeindex = {}
    user_to_timestamps = {}
    for uid, construct_event_sequence in clean_user_construct_event_sequence.items():
        timestamps = []
        for cid, event_sequence in construct_event_sequence.items():
            for time, event in event_sequence: timestamps.append(time)
        timestamps = sorted(timestamps)
        time_to_index = {t: i for i, t in enumerate(timestamps)}
        user_to_timestamps[uid] = timestamps
        user_construct_event_sequence_timeindex[uid] = {'max_index': len(time_to_index) - 1}
        for cid, event_sequence in construct_event_sequence.items():
            for time, event in event_sequence:
                if cid not in user_construct_event_sequence_timeindex[uid]:
                    user_construct_event_sequence_timeindex[uid][cid] = [(time_to_index[time], event)]
                else:
                    user_construct_event_sequence_timeindex[uid][cid].append((time_to_index[time], event))
    with open(f'./data/inoutpair/user_construct_event_sequence_timeindex.json', 'w') as f:
        json.dump(user_construct_event_sequence_timeindex, f, indent=4)
    with open(f'./data/inoutpair/user_to_timestamps.json', 'w') as f:
        json.dump(user_to_timestamps, f, indent=4)

    ### format compact data to knowledge tracing methods for imputation
    bkt_format_data = []  # prepare for pyBKT https://github.com/CAHLR/pyBKT
    dkt_format_data = []  # prepare for DKT https://github.com/ckyeungac/deep-knowledge-tracing-plus
    users_stored_for_dkt = []
    for user in user_construct_event_sequence_timeindex:
        maxind = user_construct_event_sequence_timeindex[user]['max_index']
        user_constructs_sequence, user_correct_sequence = [0 for _ in range(maxind + 1)], [0 for _ in range(maxind + 1)]
        for cid, time_events in user_construct_event_sequence_timeindex[user].items():
            if cid == 'max_index': continue
            for time, event in time_events:
                user_constructs_sequence[time] = constructs_to_idx[int(cid)]
                user_correct_sequence[time] = BINARY_SCORE_rubric[event]
        dkt_format_data.append((user_constructs_sequence, user_correct_sequence))
        users_stored_for_dkt.append(int(user))
        for cid, correct in zip(user_constructs_sequence, user_correct_sequence): bkt_format_data.append([int(user), cid, correct])

    bkt_format_data = pd.DataFrame(bkt_format_data, columns=['user_id', 'skill_name', 'correct'], dtype=int)
    bkt_format_data.to_csv('./data/inoutpair/bkt_format_data.csv', index=False)

    whole_str = ''''''
    for user_constructs_sequence, user_correct_sequence in dkt_format_data:
        whole_str += f'{len(user_constructs_sequence)}\n'
        whole_str += ','.join([str(cid) for cid in user_constructs_sequence]) + '\n'
        whole_str += ','.join([str(correct) for correct in user_correct_sequence]) + '\n'
    with open('./data/inoutpair/dkt_format_data.csv', 'w') as f:
        f.write(whole_str)

    with open('./data/inoutpair/dkt_stored_users.json', 'w') as f:
        json.dump(users_stored_for_dkt, f)

def parse_data_per_timestamp():
    # don't care about the exact event tuples (e.g., in/out, or lesson). view each timestamp as a single event.
    # the data are then formatted for some knowledge tracing methods for imputation.
    # in this case, we have more data points, though the questions inside each pair might be messy.
    os.makedirs('./data/eachstamp', exist_ok=True)
    bkt_format_data = [] # prepare for pyBKT https://github.com/CAHLR/pyBKT
    dkt_format_data = [] # prepare for DKT https://github.com/ckyeungac/deep-knowledge-tracing-plus
    users_stored_for_dkt = []

    proc_data_user_group = [df for _, df in final_loaded_data.groupby("UserId", as_index=False)]
    for user_data in proc_data_user_group:
        cur_user = user_data["UserId"].iloc[0]
        user_constructs_sequence, user_correct_sequence = [], []
        for _, row in user_data.iterrows():
            cid = constructs_to_idx[int(row['ConstructId'])]
            correct = row['IsCorrect'] if row['Type'] != 'Lesson' else 1
            bkt_format_data.append([cur_user, cid, correct])
            user_constructs_sequence.append(cid)
            user_correct_sequence.append(int(correct))
        dkt_format_data.append((user_constructs_sequence, user_correct_sequence))
        users_stored_for_dkt.append(int(cur_user))

    bkt_format_data = pd.DataFrame(bkt_format_data, columns=['user_id', 'skill_name', 'correct'], dtype=int)
    bkt_format_data.to_csv('./data/eachstamp/bkt_format_data.csv', index=False)

    whole_str = ''''''
    for user_constructs_sequence, user_correct_sequence in dkt_format_data:
        whole_str += f'{len(user_constructs_sequence)}\n'
        whole_str += ','.join([str(cid) for cid in user_constructs_sequence]) + '\n'
        whole_str += ','.join([str(correct) for correct in user_correct_sequence]) + '\n'
    with open('./data/eachstamp/dkt_format_data.csv', 'w') as f: f.write(whole_str)

    with open('./data/eachstamp/dkt_stored_users.json', 'w') as f:
        json.dump(users_stored_for_dkt, f)

if __name__ == '__main__':
    final_loaded_data = load_original_data()
    parse_data_per_in_out_session()
    parse_data_per_timestamp()