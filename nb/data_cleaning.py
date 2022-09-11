import pandas as pd
import numpy as np
import re
from utils import *

def clean_disinfo(file='../data/covid19_disinfo/covid19_disinfo_binary_english_train_old.tsv'):
    disinfo = pd.read_csv(file, sep="\t")
    disinfo = disinfo[(disinfo['q1_label'] == 'yes')
                  &disinfo['q2_label'].notna()
                  &disinfo['q4_label'].notna()][['tweet_text', 'q2_label', 'q4_label']]

    disinfo['q2_label'] = label_encode(disinfo['q2_label'])
    disinfo['q4_label'] = label_encode(disinfo['q4_label'])
    disinfo['tidy_tweet'] = clean_tweets(disinfo['tweet_text'])
    
    return disinfo


def clean_infodemic(file='../data/covid19_infordemic/covid19_infodemic_english_data.tsv'):
    infodemic = pd.read_csv(file, sep = '\t')
    infodemic = infodemic[(infodemic['q1_label'] == 'yes')
                  &(infodemic['q2_label']!='3_not_sure')
                  &(infodemic['q4_label']!='3_not_sure')][['text', 'q2_label', 'q4_label']]

    infodemic['q2_label'] = label_encode(infodemic['q2_label'])
    infodemic['q4_label'] = label_encode(infodemic['q4_label'])
    infodemic = infodemic.rename(columns={'text': 'tweet_text'})
    infodemic['tidy_tweet'] = clean_tweets(infodemic['tweet_text'])
    
    return infodemic


def clean_manual_data():
    manual_labels = ['APFactCheck_label.csv', 'BandyGit_label.csv', 'hodgetwins_label.csv', 'MackayIM_label.csv', 
    'realDennisLynch_label.csv', 'Rustybe_label.csv', 'VPrasadMD_label.csv'] 

    manual_labeled_data = pd.DataFrame()
    for f in manual_labels:
        temp = pd.read_csv('../data/manual_label/'+f)
        manual_labeled_data = manual_labeled_data.append(temp)

    manual_labeled_data = manual_labeled_data[(manual_labeled_data['q0_label'] == 'yes')
                                             &(manual_labeled_data['q1_label'] == 'yes')
                                             &manual_labeled_data['q2_label'].notna()
                                             &manual_labeled_data['q4_label'].notna()][['Tweet', 'q2_label', 'q4_label']]
    manual_labeled_data['q2_label'] = label_encode(manual_labeled_data['q2_label'])
    manual_labeled_data['q4_label'] = label_encode(manual_labeled_data['q4_label'])
    manual_labeled_data['tidy_tweet'] = clean_tweets(manual_labeled_data['Tweet'])
    manual_labeled_data = manual_labeled_data.rename(columns={'Tweet': 'tweet_text'})
    
    return manual_labeled_data
    
    
def clean_constraint_competition(file='../data/CONSTRAINT_competition/Constraint_Train.csv'):
    constraint = pd.read_csv(file)[['tweet', 'label']]
    constraint = constraint.rename(columns={'tweet': 'tweet_text', 'label': 'q2_label'})
    constraint['q2_label'] = label_encode(constraint['q2_label'])
    constraint['tidy_tweet'] = clean_tweets(constraint['tweet_text'])
    
    return constraint
    

def clean_cross_sean(file='../data/Cross_SEAN/train.txt'):
    cross_sean = pd.read_table(file, header=None)
    cross_sean = cross_sean.rename(columns={0: 'q2_label', 1: 'tweet_text'})
    cross_sean['tidy_tweet'] = clean_tweets(cross_sean['tweet_text'])

    return cross_sean
      