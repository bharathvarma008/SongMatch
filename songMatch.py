
# coding: utf-8

# author: Bharath varma
# Organisation: mtw labs innovations pvt ltd.


from __future__ import division

import os
import glob
import argparse
import pickle
import numpy as np
import pandas as pd
from statistics import mean, stdev

import librosa
import pydub #to manipulate audio
from preprocess import convert_to_ogg, get_beat_features


folder = glob.glob(os.getcwd() + '/all_songs/*')

def comp_feature_distr(dstr_array1, dstr_array2, threshold=0.01):
    
    #if not (dstr_array1.shape == dstr_array2.shape):
    #    return "Features are not compatible"
    #else:
    m = len(dstr_array1)
    list_ = []
    for i in range(m):
        mean_diff = dstr_array1[i][0] - dstr_array2[i][0]
        std_diff = dstr_array1[i][1] - dstr_array2[i][1]
        if (mean_diff <= threshold) & (std_diff <= threshold):
            list_.append(1)
        else:
            list_.append(0)
    score = len([item for item in list_ if item==1])/m
    #print list_
    return score

def get_results(test_file_name, feat_folder_list, min_perc = 0.60):
    matchList = []
    scrList = []
    # get featList from the test_file_name
    testFile_feat = get_beat_features(test_file_name)
    
    # Iterate over the featList to compare with those songs
    for featItem in feat_folder_list:
        with open(featItem, "rb") as fn:
            bb = pickle.load(fn)
        score = comp_feature_distr(bb, testFile_feat)
        scrList.append(score)

        if score >= min_perc:
            matchList.append('Yes')
        else:
            matchList.append('No')
    songList = [item.split('/')[-1][:-4] for item in feat_folder_list]
    df = pd.DataFrame({'SongName': songList,
                       'score': scrList,
                       'MatchCase': matchList})
    return df

#%time
#featList_folder = glob.glob(os.getcwd() + '/featList/*')
#get_results('test.mp3', featList_folder, min_perc=0.70)

