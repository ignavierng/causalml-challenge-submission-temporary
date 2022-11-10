#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
# from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import json
from numpyencoder import NumpyEncoder

def isOneToOne(df, col1, col2):
    return df.groupby(col1)[col2].apply(lambda x: x.nunique() == 1).all()

def isFunction(df, col1, col2, allow_missing=False):
    for qid in df[col1].unique():
        if not df[df[col1] == qid][col2].nunique() == 1:
            if not allow_missing or not df[df[col1] == qid][col2].isna().all():
                print(qid, df[df[col1] == qid][col2].unique())
                return False # print(qid, df[df['ConstructId'] == qid]['QuizId'].unique())
    return True

def check_topic_path():
    # There are 552 unique QuizId.
    # There are 2653 unique CheckinQuestionId.
    # There are 2653 unique CheckoutQuestionId.
    # Checkin and Checkout QuestionIds are one-to-one.
    # There are 1041 unique ConstructId.
    # There are 186 unique SubjectId.

    # QuizId and QuizSequence is one-to-one.
    # A question can either be as a checkin question or a checkout one, but not both.
    # A question may appear in multiple quizzes, e.g., 75910 in [242820 202553 255532].
    # Each question belongs to only one construct.
    #      and (thus) one construct may appear in multiple quizzes.
    # Each construct belongs to only one subject, or
    #      a construct may not belong to any subject, e.g., 3443 in line 50 (data missing, very few).
    # A question may also be linked to multiple subjects (maybe indirect relationship),
    #      and note that a question's own subject may not be in the QuestionSubjectIds (around 19% of questions).

    # Note that a same subject may have multiple levels and year groups. E.g., 200: [ 1  3 26 31 37 46 55], [5, 7, 8, 9].
    # So does a same construct. E.g., 47: [1 11], [5 6].
    # And even does a same question. E.g., 75910: [ 1 15 66], [5, 7, 9].
    # Only for a same quiz, its level and year group are the same (i.e., designed for each quiz).


    topic_path = '../../../../data/Task_3_dataset/topic_pathway_metadata.csv'
    df = pd.read_csv(topic_path)

    for nm in ['QuizId', 'CheckinQuestionId', 'CheckoutQuestionId', 'ConstructId', 'SubjectId']:
        print(f'There are {df[nm].nunique()} unique {nm}.')

    assert isOneToOne(df, 'CheckinQuestionId', 'CheckoutQuestionId')
    assert isOneToOne(df, 'QuizId', 'QuizSequence')
    assert isFunction(df, 'CheckinQuestionId', 'ConstructId') # so is checkout
    assert isFunction(df, 'ConstructId', 'SubjectId', allow_missing=True)
    assert isFunction(df, 'QuizId', 'Level') # so is checkout
    assert isFunction(df, 'QuizId', 'YearGroup') # so is checkout

    # lll = []
    # for _, row in df.iterrows():
    #     lst = eval(row['QuestionSubjectIds'])
    #     if pd.isna(row['SubjectId']): continue
    #     lll.append(int(row['SubjectId']) in lst)
    # print(np.mean(lll))

def check_constructs_input():
    # There are 116 constructs, over which we want to discover a graph.
    #   All constructs can be found in the topic_pathway_metadata, except for the id 469.
    #   Construct 469 appears in the training data, but its respective questions are missing in topic_pathway_metadata.

    # num_of_row_occurances_in_data: [161, 36, 744, 414, 459, 277, 141, 20, ...
    # num_of_quiz_session_occurances_in_data: [113, 13, 442, 339, 74, 171, 70, ...
    # num_of_student_learned_this_in_data: [92, 13, 395, 315, 74, 161, 62, 9,...

    # For one student, after accessing one construct, how likely will he/she revisit this construct later in another quiz?
    #    (1. 'accessing' means answering a question of this construct in a quiz,
    #     2. just like what we have in task1, so we can view the time series of this construct on the student).
    # However, this generally doesn't happen:
    #     '''
    #     for cid in constructs:
    #         students_who_learned_this_construct = data_df[data_df['ConstructId'] == cid]['UserId'].unique()
    #         learned_times = []
    #         for sid in students_who_learned_this_construct:
    #             sc_rows = data_df[(data_df['ConstructId'] == cid) & (data_df['UserId'] == sid)]
    #             sessions_of_this_student = sc_rows['QuizSessionId'].unique()
    #             learned_times.append(len(sessions_of_this_student))
    #         print(cid, np.mean(learned_times), np.median(learned_times))
    #     '''
    #     The result is like: 1272 1.2282608695652173 1.0; 1795 1.0 1.0;..., which means that
    # Generally, a student may only access a construct once, without revisiting it in later quizzes. (though there are few exceptions)
    # Then, what if not revisit a same construct in "later quizzes", but inside the same quiz?
    #     '''
    #     num_questions_per_quiz = []
    #     num_constructs_per_quiz = []
    #     for qid in data_df['QuizSessionId'].unique():
    #         checkin_constructs = data_df[(data_df['QuizSessionId'] == qid) & (data_df['Type'] == 'Checkin')]['ConstructId']
    #         num_questions_per_quiz.append(len(checkin_constructs)) # with duplicates
    #         num_constructs_per_quiz.append(checkin_constructs.nunique())
    #     print(np.mean(num_questions_per_quiz), np.median(num_questions_per_quiz))
    #     print(np.mean(num_constructs_per_quiz), np.median(num_constructs_per_quiz))
    #     '''
    #     The result is around (early stopped at 1000 sessions): (4.674, 5.0), (3.51, 4.0), which means that
    # For the around 5 questions in a quiz, there might exist a revisit to a same construct (by different questions)/ (~2 constructs per quiz).
    # We may use this temporal information later;
    #     also, except for revisiting by different questions, we may consider the learning lessons and/or retry of a same question.

    data_path = '../../../../data/Task_3_dataset/checkins_lessons_checkouts_training.csv'
    data_df = pd.read_csv(data_path)
    topic_path = '../../../../data/Task_3_dataset/topic_pathway_metadata.csv'
    topic_df = pd.read_csv(topic_path)
    subject_path = '../../../../data/Task_3_dataset/subject_metadata.csv'
    subject_df = pd.read_csv(subject_path)
    users_path = '../../../../data/Task_3_dataset/student_metadata.csv'
    users_df = pd.read_csv(users_path)
    input_test_path = '../../../../data/Task_3_dataset/constructs_input_test.csv'
    constructs = pd.read_csv(input_test_path).to_numpy()[:, 0]

    ####### save more info to csv file #######
    respective_subject_id = [int(topic_df[topic_df['ConstructId'] == cid]['SubjectId'].unique()[0])
                             if cid != 469 else np.nan for cid in constructs]  # 469 not occurring in subject_df
    respective_subject_name = [subject_df[subject_df['SubjectId'] == sid]['Name'].unique()[0]
                               if not np.isnan(sid) else "" for sid in respective_subject_id]
    respective_subject_parent2 = [int(subject_df[subject_df['SubjectId'] == sid]['ParentId'].unique()[0])
                                  if not np.isnan(sid) else np.nan for sid in respective_subject_id]
    respective_subject_parent2_name = [subject_df[subject_df['SubjectId'] == sid]['Name'].unique()[0]
                                       if not np.isnan(sid) else "" for sid in respective_subject_parent2]
    respective_subject_parent1 = [int(subject_df[subject_df['SubjectId'] == sid]['ParentId'].unique()[0])
                                  if not np.isnan(sid) else np.nan for sid in respective_subject_parent2]
    respective_subject_parent1_name = [subject_df[subject_df['SubjectId'] == sid]['Name'].unique()[0]
                                       if not np.isnan(sid) else "" for sid in respective_subject_parent1]

    respective_subject_level = [subject_df[subject_df['SubjectId'] == sid]['Level'].unique()[0]
                                if not np.isnan(sid) else np.nan for sid in respective_subject_id]  # all is 3 (leaf)

    quiz_sessions_containing_this_in_data = [data_df[data_df['ConstructId'] == cid]["QuizSessionId"].unique() for cid in constructs]
    quizzes_containing_this_in_data = [data_df[data_df['ConstructId'] == cid]["QuizId"].unique() for cid in constructs]
    questions_containing_this_in_data = [data_df[data_df['ConstructId'] == cid]["QuestionId"].unique() for cid in constructs]
    students_accessing_this_in_data = [data_df[data_df['ConstructId'] == cid]["UserId"].unique() for cid in constructs]

    quizzes_containing_this_in_topic_path = [topic_df[topic_df['ConstructId'] == cid]["QuizId"].unique() for cid in constructs]
    checkin_questions_containing_this_in_topic_path = [
        topic_df[topic_df['ConstructId'] == cid]["CheckinQuestionId"].unique() for cid in constructs]
    checkout_questions_containing_this_in_topic_path = [
        topic_df[topic_df['ConstructId'] == cid]["CheckoutQuestionId"].unique() for cid in constructs]
    levels_of_quizzes_containing_this_in_topic_path = [
        topic_df[topic_df['ConstructId'] == cid]["Level"].unique() for cid in constructs]
    year_groups_of_quizzes_containing_this_in_topic_path = [
        topic_df[topic_df['ConstructId'] == cid]["YearGroup"].unique() for cid in constructs]
    sequences_of_quizzes_containing_this_in_topic_path = [
        topic_df[topic_df['ConstructId'] == cid]["QuizSequence"].unique() for cid in constructs]

    num_quiz_sessions_containing_this_in_data = list(map(len, quiz_sessions_containing_this_in_data))
    num_quizzes_containing_this_in_data = list(map(len, quizzes_containing_this_in_data))
    num_questions_containing_this_in_data = list(map(len, questions_containing_this_in_data))
    num_students_accessing_this_in_data = list(map(len, students_accessing_this_in_data))
    num_quizzes_containing_this_in_topic_path = list(map(len, quizzes_containing_this_in_topic_path))
    num_checkin_questions_containing_this_in_topic_path = list(map(len, checkin_questions_containing_this_in_topic_path))
    num_checkout_questions_containing_this_in_topic_path = list(map(len, checkout_questions_containing_this_in_topic_path))

    stringlize_arr = lambda arr: ",".join(map(str, arr))

    input_constructs_with_more_info_df = pd.DataFrame({
        'ConstructId': constructs,
        'SubjectId': respective_subject_id,
        'SubjectName': respective_subject_name,
        'SubjectParentLevel2': respective_subject_parent2,
        'SubjectParentLevel2Name': respective_subject_parent2_name,
        'SubjectParentLevel1': respective_subject_parent1,
        'SubjectParentLevel1Name': respective_subject_parent1_name,
        'SubjectLevel': respective_subject_level,
        'LevelsOfQuizzesContainingThisInTopicPath': list(map(stringlize_arr, levels_of_quizzes_containing_this_in_topic_path)),
        'YearGroupsOfQuizzesContainingThisInTopicPath': list(map(stringlize_arr, year_groups_of_quizzes_containing_this_in_topic_path)),
        'SequencesOfQuizzesContainingThisInTopicPath': list(map(stringlize_arr, sequences_of_quizzes_containing_this_in_topic_path)),
        'NumQuizSessionsContainingThisInData': num_quiz_sessions_containing_this_in_data,
        'NumQuizzesContainingThisInData': num_quizzes_containing_this_in_data,
        'NumQuestionsContainingThisInData': num_questions_containing_this_in_data,
        'NumStudentsAccessingThisInData': num_students_accessing_this_in_data,
        'NumQuizzesContainingThisInTopicPath': num_quizzes_containing_this_in_topic_path,
        'NumCheckinQuestionsContainingThisInTopicPath': num_checkin_questions_containing_this_in_topic_path,
        'NumCheckoutQuestionsContainingThisInTopicPath': num_checkout_questions_containing_this_in_topic_path,
        'QuizSessionsContainingThisInData': list(map(stringlize_arr, quiz_sessions_containing_this_in_data)),
        'QuizzesContainingThisInData': list(map(stringlize_arr, quizzes_containing_this_in_data)),
        'QuestionsContainingThisInData': list(map(stringlize_arr, questions_containing_this_in_data)),
        'StudentsAccessingThisInData': list(map(stringlize_arr, students_accessing_this_in_data)),
        'QuizzesContainingThisInTopicPath': list(map(stringlize_arr, quizzes_containing_this_in_topic_path)),
        'CheckinQuestionsContainingThisInTopicPath': list(map(stringlize_arr, checkin_questions_containing_this_in_topic_path)),
        'CheckoutQuestionsContainingThisInTopicPath': list(map(stringlize_arr, checkout_questions_containing_this_in_topic_path)),
    })

    input_constructs_with_more_info_df.to_csv('../../../../data/Task_3_dataset/constructs_input_test_more_info.csv', index=False)

    ####### save more info to json file #######
    ALLRES = {}
    for cid in constructs:
        subject_id = int(topic_df[topic_df['ConstructId'] == cid]['SubjectId'].unique()[0]) if cid != 469 else None  # 469 not occurring in subject_df
        subject_name = subject_df[subject_df['SubjectId'] == subject_id]['Name'].unique()[0] if subject_id is not None else ""
        subject_parent2 = int(subject_df[subject_df['SubjectId'] == subject_id]['ParentId'].unique()[0]) if subject_id is not None else None
        subject_parent2_name = subject_df[subject_df['SubjectId'] == subject_parent2]['Name'].unique()[0] if subject_parent2 is not None else ""
        subject_parent1 = int(subject_df[subject_df['SubjectId'] == subject_parent2]['ParentId'].unique()[0]) if subject_parent2 is not None else None
        subject_parent1_name = subject_df[subject_df['SubjectId'] == subject_parent1]['Name'].unique()[0] if subject_parent1 is not None else ""
        subject_level = subject_df[subject_df['SubjectId'] == subject_id]['Level'].unique()[0] if subject_id is not None else None # all is 3 (leaf)

        quiz_sessions_info_containing_this_in_data = []
        dfc = data_df[data_df['ConstructId'] == cid]
        for row in dfc.iterrows():
            user_id, quiz_session_id, question_sequence, question_id, type, timestamp, iscorrect = \
                row[1]['UserId'], row[1]['QuizSessionId'], row[1]['QuestionSequence'], \
                row[1]['QuestionId'], row[1]['Type'], row[1]['Timestamp'], row[1]['IsCorrect']
            dfu = users_df[users_df['UserId'] == user_id]
            user_birth_date, user_year_group = dfu['MonthOfBirth'].unique()[0], dfu['YearGroup'].unique()[0]
            age_now = str(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f") - datetime.strptime(user_birth_date, "%Y-%m-%d")) \
                if isinstance(user_birth_date, str) else ""
            quiz_sessions_info_containing_this_in_data.append(
                {'user_id': int(user_id), 'age_now': age_now, 'user_year_group': int(user_year_group),
                 'quiz_session_id': int(quiz_session_id), 'question_sequence': int(question_sequence),
                 'question_id': int(question_id), 'type': type, 'timestamp': timestamp, 'iscorrect': iscorrect if not np.isnan(iscorrect) else None})

        quiz_subject_info_containing_this_in_meta = []
        dfq = topic_df[topic_df['ConstructId'] == cid]
        for row in dfq.iterrows():
            level, year_group, quiz_sequence, question_sequence,\
            quiz_id, checkin_qid, checkout_qid, subject_id, question_subject_ids = \
                row[1]['Level'], row[1]['YearGroup'], row[1]['QuizSequence'], row[1]['QuestionSequence'], \
                row[1]['QuizId'], row[1]['CheckinQuestionId'], row[1]['CheckoutQuestionId'], \
                row[1]['SubjectId'], eval('[' + row[1]['QuestionSubjectIds'] + ']')
            quiz_subject_info_containing_this_in_meta.append(
                {'level': int(level), 'year_group': int(year_group), 'quiz_sequence': int(quiz_sequence),
                 'question_sequence': int(question_sequence), 'quiz_id': int(quiz_id),
                 'checkin_qid': int(checkin_qid), 'checkout_qid': int(checkout_qid),
                 'subject_id': int(subject_id), 'question_subject_ids': question_subject_ids})

        ALLRES[int(cid)] = {'subject_id': subject_id, 'subject_name': subject_name,
                       'subject_parent2': subject_parent2, 'subject_parent2_name': subject_parent2_name,
                       'subject_parent1': subject_parent1, 'subject_parent1_name': subject_parent1_name, 'subject_level': subject_level,
                       'quiz_sessions_info_containing_this_in_data': quiz_sessions_info_containing_this_in_data,
                       'quiz_subject_info_containing_this_in_meta': quiz_subject_info_containing_this_in_meta}

    with open('../../../../data/Task_3_dataset/constructs_input_test_more_info.json', 'w') as f:
        json.dump(ALLRES, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


def generate_student_row_table():
    data_path = '../../../../data/Task_3_dataset/checkins_lessons_checkouts_training.csv'
    data_df = pd.read_csv(data_path)
    input_test_path = '../../../../data/Task_3_dataset/constructs_input_test.csv'
    constructs = pd.read_csv(input_test_path).to_numpy()[:, 0]

    student_accuracies = []
    for sid in tqdm(data_df['UserId'].unique()):
        student_df = data_df[data_df['UserId'] == sid]
        construct_accuracy = [np.nanmean(student_df[student_df['ConstructId'] == cid]['IsCorrect']) for cid in constructs]
        if not np.isnan(construct_accuracy).all():
            student_accuracies.append([sid] + construct_accuracy)

    student_accuracies_df = pd.DataFrame(student_accuracies, columns=['UserId'] + list(constructs))
    student_accuracies_df.to_csv('../../../../data/Task_3_dataset/student_row_accuracies.csv', index=False)

def plot_subject_meta():
    subject_path = '../../../../data/Task_3_dataset/subject_metadata.csv'
    subject_df = pd.read_csv(subject_path)

    id_to_nodename = {}
    ids, edges = [], []
    for row in subject_df.iterrows():
        row = row[1]
        rid = int(row['SubjectId'])
        nodename = f'{row["Level"]}-{rid}\n{row["Name"]}'
        id_to_nodename[rid] = nodename
        parent = row['ParentId']
        if pd.isna(parent): continue
        edges.append((int(parent), rid))

    G = nx.DiGraph()
    G.add_edges_from(edges)
    G.add_nodes_from(ids)
    pos = graphviz_layout(G, prog='dot')

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(G, pos)
    text = nx.draw_networkx_labels(G, pos, labels=id_to_nodename, font_size=8)
    for _, t in text.items(): t.set_rotation('vertical')

    fig = plt.gcf()
    fig.set_size_inches(300, 15)

    plt.title(f'Subject Meta')
    plt.axis('off')
    plt.tight_layout(pad=0.5)
    plt.savefig(f'../../../../data/Task_3_dataset/plots/subject_meta.pdf')
    plt.cla()



if __name__ == '__main__':
    print('This file collects meta info on constructs. should take ~30 secs')
    check_topic_path()
    check_constructs_input()