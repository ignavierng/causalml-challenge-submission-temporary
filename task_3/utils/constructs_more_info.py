#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

nan = np.nan
more_info_pth = '../../../../data/Task_3_dataset/constructs_input_test_more_info.csv'
c_more_info_df = pd.read_csv(more_info_pth)
list_from_str = lambda s: eval('[' + str(s) + ']')
int_ = lambda s: int(s) if not np.isnan(s) else nan

constructs = c_more_info_df['ConstructId'].values
respective_subject_id = c_more_info_df['SubjectId'].values
respective_subject_name = c_more_info_df['SubjectName'].values
respective_subject_parent2 = c_more_info_df['SubjectParentLevel2'].values
respective_subject_parent2_name = c_more_info_df['SubjectParentLevel2Name'].values
respective_subject_parent1 = c_more_info_df['SubjectParentLevel1'].values
respective_subject_parent1_name = c_more_info_df['SubjectParentLevel1Name'].values
respective_subject_level = c_more_info_df['SubjectLevel'].values
levels_of_quizzes_containing_this_in_topic_path = list(map(list_from_str, c_more_info_df['LevelsOfQuizzesContainingThisInTopicPath'].values))
year_groups_of_quizzes_containing_this_in_topic_path = list(map(list_from_str, c_more_info_df['YearGroupsOfQuizzesContainingThisInTopicPath'].values))
sequences_of_quizzes_containing_this_in_topic_path = list(map(list_from_str, c_more_info_df['SequencesOfQuizzesContainingThisInTopicPath'].values))
num_quiz_sessions_containing_this_in_data = c_more_info_df['NumQuizSessionsContainingThisInData'].values
num_quizzes_containing_this_in_data = c_more_info_df['NumQuizzesContainingThisInData'].values
num_questions_containing_this_in_data = c_more_info_df['NumQuestionsContainingThisInData'].values
num_students_accessing_this_in_data = c_more_info_df['NumStudentsAccessingThisInData'].values
num_quizzes_containing_this_in_topic_path = c_more_info_df['NumQuizzesContainingThisInTopicPath'].values
num_checkin_questions_containing_this_in_topic_path = c_more_info_df['NumCheckinQuestionsContainingThisInTopicPath'].values
num_checkout_questions_containing_this_in_topic_path = c_more_info_df['NumCheckoutQuestionsContainingThisInTopicPath'].values
quiz_sessions_containing_this_in_data = list(map(list_from_str, c_more_info_df['QuizSessionsContainingThisInData'].values))
quizzes_containing_this_in_data = list(map(list_from_str, c_more_info_df['QuizzesContainingThisInData'].values))
questions_containing_this_in_data = list(map(list_from_str, c_more_info_df['QuestionsContainingThisInData'].values))
students_accessing_this_in_data = list(map(list_from_str, c_more_info_df['StudentsAccessingThisInData'].values))
quizzes_containing_this_in_topic_path = list(map(list_from_str, c_more_info_df['QuizzesContainingThisInTopicPath'].values))
checkin_questions_containing_this_in_topic_path = list(map(list_from_str, c_more_info_df['CheckinQuestionsContainingThisInTopicPath'].values))
checkout_questions_containing_this_in_topic_path = list(map(list_from_str, c_more_info_df['CheckoutQuestionsContainingThisInTopicPath'].values))

# now we want to create a dictionary of dictionaries
# the first key is the construct id
# the second key is the name of the field
# the value is the value of the field
c_more_info_dict = {}
for i in range(len(constructs)):
    c_more_info_dict[constructs[i]] = {
        'SubjectId': int_(respective_subject_id[i]),
        'SubjectName': respective_subject_name[i],
        'SubjectParentLevel2': int_(respective_subject_parent2[i]),
        'SubjectParentLevel2Name': respective_subject_parent2_name[i],
        'SubjectParentLevel1': int_(respective_subject_parent1[i]),
        'SubjectParentLevel1Name': respective_subject_parent1_name[i],
        'SubjectLevel': int_(respective_subject_level[i]),
        'LevelsOfQuizzesContainingThisInTopicPath': levels_of_quizzes_containing_this_in_topic_path[i],
        'YearGroupsOfQuizzesContainingThisInTopicPath': year_groups_of_quizzes_containing_this_in_topic_path[i],
        'SequencesOfQuizzesContainingThisInTopicPath': sequences_of_quizzes_containing_this_in_topic_path[i],
        'NumQuizSessionsContainingThisInData': num_quiz_sessions_containing_this_in_data[i],
        'NumQuizzesContainingThisInData': num_quizzes_containing_this_in_data[i],
        'NumQuestionsContainingThisInData': num_questions_containing_this_in_data[i],
        'NumStudentsAccessingThisInData': num_students_accessing_this_in_data[i],
        'NumQuizzesContainingThisInTopicPath': num_quizzes_containing_this_in_topic_path[i],
        'NumCheckinQuestionsContainingThisInTopicPath': num_checkin_questions_containing_this_in_topic_path[i],
        'NumCheckoutQuestionsContainingThisInTopicPath': num_checkout_questions_containing_this_in_topic_path[i],
        'QuizSessionsContainingThisInData': quiz_sessions_containing_this_in_data[i],
        'QuizzesContainingThisInData': quizzes_containing_this_in_data[i],
        'QuestionsContainingThisInData': questions_containing_this_in_data[i],
        'StudentsAccessingThisInData': students_accessing_this_in_data[i],
        'QuizzesContainingThisInTopicPath': quizzes_containing_this_in_topic_path[i],
        'CheckinQuestionsContainingThisInTopicPath': checkin_questions_containing_this_in_topic_path[i],
        'CheckoutQuestionsContainingThisInTopicPath': checkout_questions_containing_this_in_topic_path[i],
    }

import json
more_info_pth = '../../../../data/Task_3_dataset/constructs_input_test_more_info.json'
with open(more_info_pth, 'r') as f: new_more_info = json.load(f)
new_more_info = {int(k): v for k, v in new_more_info.items()}

for k, v in new_more_info.items():
    c_more_info_dict[k]["quiz_sessions_info_containing_this_in_data"] = v["quiz_sessions_info_containing_this_in_data"]
    c_more_info_dict[k]["quiz_subject_info_containing_this_in_meta"] = v["quiz_subject_info_containing_this_in_meta"]


