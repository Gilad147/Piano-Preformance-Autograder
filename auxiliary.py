import math

import pandas as pd

import Performance_class
import Song_Class
import numpy as np
import pretty_midi
import os
import shutil
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

import pickle


def change_midi_file_tempo(original_path, new_path, percentage=0.10):
    performance = Performance_class.Performance(original_path, " ", " ",
                                                original_path)
    percentage = -percentage
    if percentage > 0:
        performance.mistakes_generator("rhythm", noise=percentage)
        performance.mistakes_generator("duration", noise=percentage, original=False)
    else:
        performance.mistakes_generator("duration", noise=percentage)
        performance.mistakes_generator("rhythm", noise=percentage, original=False)
    np2mid(performance.midi_df, new_path, None, True)
    return new_path


def np2mid(np_performance, midfilename, original_midi_file, write_midi_file):
    """
    Converts an numpy array  to a .mid file

    @param np_performance: np array with Midi values
    @param midfilename: full path to the mid output file
    @return: None
    """

    performance = pretty_midi.PrettyMIDI()

    piano = pretty_midi.Instrument(program=4)
    # Iterate over note names, which will be converted to note number later
    for m in np_performance:
        note = pretty_midi.Note(velocity=int(m[3]), pitch=int(m[2]), start=m[0], end=m[1])
        piano.notes.append(note)
    performance.instruments.append(piano)
    if write_midi_file:
        performance.write(midfilename)
    else:
        return_performance = Performance_class.Performance(path=None, name=midfilename, player_name="np2mid",
                                                           original_path=None,
                                                           prettyMidiFile_performance=performance,
                                                           prettyMidiFile_original=original_midi_file)
        return return_performance
