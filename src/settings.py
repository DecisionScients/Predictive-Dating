import os
HOME_DIR = r"c:\Users\John\Documents\Data Science\Projects\Predictive Dating"
RAW_DATA_DIR = "./data/raw"
INTERIM_DATA_DIR = "./data/interim"
PROCESSED_DATA_DIR = "./data/processed"
RAW_DATA_FILENAME = 'speed-dating_csv.csv'
VARS = ['gender', 'age', 'age_o', 'd_age', 'race', 'race_o', 'samerace',
        'importance_same_race', 'importance_same_religion',
        'pref_o_attractive', 'pref_o_sincere', 'pref_o_intelligence',
        'pref_o_funny', 'pref_o_ambitious', 'pref_o_shared_interests',
        'attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o',
        'ambitous_o', 'shared_interests_o', 'attractive_important',
        'sincere_important', 'intellicence_important',
        'funny_important', 'ambtition_important',
        'shared_interests_important', 'attractive',	'sincere',
        'intelligence', 'funny', 'ambition', 'attractive_partner',
        'sincere_partner', 'intelligence_partner', 'funny_partner',
        'ambition_partner', 'shared_interests_partner',
        'interests_correlate', 'expected_happy_with_sd_people',
        'guess_prob_liked', 'decision', 'decision_o', 'match']
