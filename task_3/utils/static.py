#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pandas as pd
input_test_path = '../../../../data/Task_3_dataset/constructs_input_test.csv'
constructs = pd.read_csv(input_test_path).to_numpy()[:, 0] # name of the constructs
constructs_to_idx = {constructs[i]: i for i in range(len(constructs))}

EVENT_TO_EXACT_SCORE_rubric = {
    'Checkin correct': 100,
    'Checkin wrong, CheckinRetry correct': 70,
    'Checkin wrong, CheckinRetry wrong, Lesson, Checkout correct': 50,
    'Checkin wrong, CheckinRetry wrong, Lesson, Checkout wrong, CheckoutRetry wrong': 0,
    'Checkin wrong, CheckinRetry wrong, Lesson, Checkout wrong, CheckoutRetry correct': 20,
    'Checkin wrong, CheckinRetry wrong': 10,
    'Checkin correct, Lesson, Checkout correct': 90,
    'Checkin wrong, CheckinRetry wrong, Checkout wrong, CheckoutRetry wrong': 0,
    'Checkin wrong, CheckinRetry wrong, Checkout correct': 50,
    'Checkin wrong, CheckinRetry wrong, Checkout wrong, CheckoutRetry correct': 30,
    'Checkin wrong, CheckinRetry correct, Lesson, Checkout correct': 60,
    'Checkin wrong': 20,
    'Checkin correct, Lesson, Checkout wrong, CheckoutRetry correct': 80,
    'Checkin correct, Checkout correct': 95,
    'Checkin correct, Lesson, Checkout wrong, CheckoutRetry wrong': 30,
    'Checkin wrong, CheckinRetry correct, Lesson, Checkout wrong, CheckoutRetry correct': 40,
    'Checkin wrong, CheckinRetry correct, Lesson, Checkout wrong, CheckoutRetry wrong': 30,
}

EVENT_TO_GAIN_SCORE_rubric = {
    'Checkin correct': +5,
    'Checkin wrong, CheckinRetry correct': +3,
    'Checkin wrong, CheckinRetry wrong, Lesson, Checkout correct': +15, # very good. initially failed; after learning now good
    'Checkin wrong, CheckinRetry wrong, Lesson, Checkout wrong, CheckoutRetry wrong': +1, # can barely see any gain from learning
    'Checkin wrong, CheckinRetry wrong, Lesson, Checkout wrong, CheckoutRetry correct': +7, # learned. but not very mastered
    'Checkin wrong, CheckinRetry wrong': -3, # i.e., learned before, but failed again. deduct our guess of the score.
    'Checkin correct, Lesson, Checkout correct': +10, # already very good (but maybe not that sure), and also learned more
    'Checkin wrong, CheckinRetry wrong, Checkout wrong, CheckoutRetry wrong': -7, # learned before. but failed again.
    'Checkin wrong, CheckinRetry wrong, Checkout correct': +2,
    'Checkin wrong, CheckinRetry wrong, Checkout wrong, CheckoutRetry correct': +1,
    'Checkin wrong, CheckinRetry correct, Lesson, Checkout correct': +18, # very good.
    'Checkin wrong': -1,
    'Checkin correct, Lesson, Checkout wrong, CheckoutRetry correct': +10, # checkout wrong is like a typo
    'Checkin correct, Checkout correct': +7,
    'Checkin correct, Lesson, Checkout wrong, CheckoutRetry wrong': +8,
    'Checkin wrong, CheckinRetry correct, Lesson, Checkout wrong, CheckoutRetry correct': +10,
    'Checkin wrong, CheckinRetry correct, Lesson, Checkout wrong, CheckoutRetry wrong': +7,
}

BINARY_SCORE_rubric = {
    "Checkin correct": 1,
    "Checkin wrong, CheckinRetry correct": 1,
    "Checkin wrong, CheckinRetry wrong, Lesson, Checkout correct": 1,
    "Checkin wrong, CheckinRetry wrong, Lesson, Checkout wrong, CheckoutRetry wrong": 0,
    "Checkin wrong, CheckinRetry wrong, Lesson, Checkout wrong, CheckoutRetry correct": 0,
    "Checkin wrong, CheckinRetry wrong": 0,
    "Checkin correct, Lesson, Checkout correct": 1,
    "Checkin wrong, CheckinRetry wrong, Checkout wrong, CheckoutRetry wrong": 0,
    "Checkin wrong, CheckinRetry wrong, Checkout correct": 0,
    "Checkin wrong, CheckinRetry wrong, Checkout wrong, CheckoutRetry correct": 0,
    "Checkin wrong, CheckinRetry correct, Lesson, Checkout correct": 1,
    "Checkin wrong": 0,
    "Checkin correct, Lesson, Checkout wrong, CheckoutRetry correct": 1,
    "Checkin correct, Checkout correct": 1,
    "Checkin correct, Lesson, Checkout wrong, CheckoutRetry wrong": 0,
    "Checkin wrong, CheckinRetry correct, Lesson, Checkout wrong, CheckoutRetry correct": 1,
    "Checkin wrong, CheckinRetry correct, Lesson, Checkout wrong, CheckoutRetry wrong": 0,
    "Checkin wrong, CheckinRetry wrong, Checkout wrong": 0,
    "Checkin correct, CheckinRetry wrong": 0,
    "Checkin correct, Checkout wrong, CheckoutRetry wrong": 0,
    "Checkin wrong, CheckinRetry correct, Checkout wrong, CheckoutRetry wrong": 0,
    "Checkin wrong, CheckinRetry correct, Checkout correct": 1,
    "Checkin wrong, CheckinRetry wrong, Lesson, Checkout wrong": 0,
    "Checkin correct, Checkout wrong, CheckoutRetry correct": 1,
    "Checkout correct": 1,
    "Checkin wrong, CheckinRetry correct, Checkout wrong, CheckoutRetry correct": 1,
    "Checkin wrong, CheckinRetry wrong, Lesson": 0,
    "CheckinRetry correct": 1,
    "Checkout wrong, CheckoutRetry wrong": 0,
    "Checkin correct, Lesson": 1,
    "Lesson, Checkout wrong, CheckoutRetry wrong": 0,
    "Checkin correct, CheckinRetry wrong, Checkout correct": 1,
    "Checkin wrong, CheckinRetry correct, Lesson, Checkout wrong": 0,
    "Checkout wrong, CheckoutRetry correct": 1,
}

def parse_time(time_str):
    # e.g., "4294 days, 18:23:19.790000", a time delta string
    if time_str == '': return np.nan
    return int(time_str.split(' days, ')[0]) # maybe could also be more precise