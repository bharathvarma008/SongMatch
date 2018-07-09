
# coding: utf-8
# author: Bharath Varma
# organisation: mtw labs innovations pvt ltd.

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

folder = glob.glob(os.getcwd() + '/all_songs/*')

def convert_to_ogg(songname):
    sound = pydub.AudioSegment.from_file(songname)
    newname = songname[:-4] + '.ogg'
    sound.export(newname, format='ogg')
    return newname

def get_beat_features(songname):
    # Check and convert if the song format is in .ogg
    songname = convert_to_ogg(songname)
    
    # Load the example clip
    y, sr = librosa.load(songname)

    # Set the hop length; at 22050 Hz, 512 samples ~= 23ms
    hop_length = 512

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                                 sr=sr)

    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)

    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)

    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                        beat_frames)

    # Compute chroma features from the harmonic signal
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                            sr=sr)

    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = librosa.util.sync(chromagram,
                                    beat_frames,
                                    aggregate=np.median)

    # Finally, stack all beat-synchronous features together
    beat_features = np.vstack([beat_chroma, beat_mfcc_delta])
    
    # from feature distribution to mean and standard deviations of those distributions
    beat_feat_distr = [[mean(item), stdev(item)] for item in beat_features]
    
    # removing unwanted .ogg file
    os.remove(songname)
    
    return beat_feat_distr

def store_feat_of_songs(file_folder):
    for song_in_folder in file_folder:
        #print song_in_folder
        song_feat = get_beat_features(song_in_folder)
        fileName = os.getcwd() + '/featList/' + song_in_folder.split('/')[-1] + '.txt'
        with open(fileName, "wb") as fn:
            pickle.dump(song_feat, fn)
    
    featList_folder = glob.glob(os.getcwd() + '/featList/*')
    if len(file_folder)==len(featList_folder):
        return "All songs are converted and feature matrix is stored in '/featList/'. "
    else:
        return "Conversion failed. All songs are not converted."


#store_feat_of_songs(folder)

