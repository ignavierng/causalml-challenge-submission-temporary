import argparse
import copy
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pickle

def raw_load_and_process_eedi_data(data_path: str, stu_meta_path: str, save_dir: str):

    loaded_data = (
        pd.read_csv(data_path, index_col=False)
        .sort_values(["UserId", "QuizSessionId", "Timestamp"], axis=0)
        .reset_index(drop=True)
    )
    stu_meta = pd.read_csv(stu_meta_path, index_col=False)

    # remove the length<10 users
    _, idx, counts = np.unique(np.array(loaded_data["UserId"]), return_counts=True, return_index=True)
    list_idx_to_remove_nest = [list(range(ind, ind + counts[counts < 10][c])) for c, ind in enumerate(idx[counts < 10])]
    list_idx_to_remove = [v for sublist in list_idx_to_remove_nest for v in sublist]  # flatten it
    proc_loaded_data = loaded_data.drop(labels=list_idx_to_remove, axis=0).reset_index(drop=True)
    # Build construct mapping from the old construct_id to the new construct_id
    const_map = {old_id: new_id for new_id, old_id in enumerate(np.unique(proc_loaded_data["ConstructId"]))}
    # Group the user by user_id
    proc_data_user_group = [df for _, df in proc_loaded_data.groupby("UserId", as_index=False)]

    # Process the data for each user
    proc_final_data: List[float] = []
    final_is_real_obs = []
    
    print(len(const_map))
    
    for user_data in proc_data_user_group:
        cur_user = user_data["UserId"].iloc[0]
        #print(cur_user)

        cur_user_year = stu_meta[stu_meta["UserId"] == cur_user]["YearGroup"]

        cur_knowledge = {0: np.nan * np.ones(len(const_map))}
        cur_is_real_obs = {0: np.zeros(len(const_map))}

        const_id = np.unique(user_data["ConstructId"])
        cur_question_number = {
            c_id: len(np.unique(user_data[user_data["ConstructId"] == c_id]["QuestionId"])) for c_id in const_id
        }
        # iterate through each row to process the raw data
        new_time = 0
        cur_bot_list = []
        for row_id, row in user_data.iterrows():
            # get new construct_id
            cur_const_id = row["ConstructId"]
            cur_const_new_id = const_map[cur_const_id]
            if row_id == 0:
                # every row should start with a Checkin question
                assert row["Type"] == "Checkin"
                
            if row["IsCorrect"] == 1:
                # If a question is correctly answered, no learning happens and it reveals the knowledge of the associated
                # construct. Thus, update the knowledge of the current time step.
                if cur_knowledge[new_time][cur_const_new_id] == np.nan:
                    cur_knowledge[new_time][cur_const_new_id] = 1 / cur_question_number[cur_const_id]
                else:
                    cur_knowledge[new_time][cur_const_new_id] += 1 / cur_question_number[cur_const_id]
                cur_is_real_obs[new_time][cur_const_new_id] = 1
                
            elif row["IsCorrect"] == 0:
                # If a queston is incorrectly answered, either a lesson or a hint are given. Thus, learning happens.
                # However, this learning only happens after a Checkin question, since other types (e.g. Checkout, CheckinRetry,etc.)
                # teach the same construct knowledge, we aggregate them into one learning session.
                if row["Type"] == "Checkin":

                    cur_knowledge[new_time][cur_const_new_id] = 0
                    cur_is_real_obs[new_time][cur_const_new_id] = 1
            
                    cur_bot_list.append(cur_const_new_id)
                    # Generate new timestamp
                    new_time += 1
                    cur_knowledge[new_time] = copy.deepcopy(cur_knowledge[new_time - 1])
                    cur_is_real_obs[new_time] = np.zeros(len(const_map))

        # Generate the processed data
        if len(cur_bot_list) < len(cur_knowledge):
            # Make them compatible shape
            cur_bot_list.append(0)
        elif len(cur_bot_list) > len(cur_knowledge):
            raise ValueError("The knowledge dict should not be smaller than bot action list")
            
        if len(cur_bot_list) >= 10:
            # remove too short length data
            cur_data_entry = [[cur_user] + [cur_bot_list[k]] + v.tolist() + cur_is_real_obs[k].tolist() for k, v in cur_knowledge.items()]
            proc_final_data = proc_final_data + cur_data_entry
            #final_is_real_obs.append([v.tolist() for k, v in cur_is_real_obs.items()])

    # Save the npy file
    proc_final_data_np = np.array(proc_final_data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savetxt(os.path.join(save_dir, "train_raw.csv"), proc_final_data_np, delimiter=",")
    
    # save is real obs
    #f=open(os.path.join(save_dir, "final_is_real_obs.pickle"),'wb')
    #pickle.dump(final_is_real_obs, f)
    #f.close()
    
    # save const_map
    f=open(os.path.join(save_dir, "const_map.pickle"),'wb')
    pickle.dump(const_map, f)
    f.close()
    #np.save(os.path.join(save_dir, "const_map.npy"), const_map)  # type: ignore

data_path = "task_4_data_raw/checkins_lessons_checkouts_training.csv"
stu_meta_path = "task_4_data_raw/student_metadata.csv"
save_dir = "task_4_data_processed"

raw_load_and_process_eedi_data(data_path, stu_meta_path, save_dir)