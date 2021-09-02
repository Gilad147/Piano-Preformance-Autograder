import Performance_class
import numpy as np
from midiutil.MidiFile import MIDIFile
import pretty_midi
import os
import shutil
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def generate_random_mistakes_data(folder, n, create_midi_files):
    basepath = folder + '/'
    all_data = []
    if create_midi_files:
        fake_data_path = folder + ' - fake data/'
        Path(fake_data_path).mkdir(exist_ok=True)
    with os.scandir(basepath) as songs:
        for song in songs:
            song_name = song.name.split(".")[0]
            if song.is_file() and song.name != '.DS_Store':
                flawed_performances, original_midi_data = create_random_mistakes(basepath + song.name, song_name, n,
                                                                                 min_noise=0.25, max_noise=0.9,
                                                                                 min_percentage=0.4, max_percentage=1)
                if create_midi_files:
                    Path(fake_data_path + song_name).mkdir(exist_ok=True)
                    shutil.copy(basepath + song.name, fake_data_path + song_name)
                    Path(fake_data_path + song_name + '/fake performances/').mkdir(exist_ok=True)
                for i, data in enumerate(flawed_performances):
                    if create_midi_files:
                        path = fake_data_path + song_name + '/fake performances/' + song_name + str(i) + ".mid"
                        np2mid(data.midi_df, path, original_midi_file=None, write_midi_file=True)
                    else:
                        fake_data_performance = np2mid(data.midi_df, "midfilename", original_midi_data[i],
                                                       write_midi_file=False)
                        all_data.append(fake_data_performance)
    return all_data


def create_random_mistakes(path, name, n, max_noise, max_percentage, min_noise=0, min_percentage=0.5):
    flawed_performances = []
    original_midi_data = []
    for i in range(n):
        performance = Performance_class.Performance(path, name, "com", path)
        original_midi_data.append(performance.midi_data_original)
        performance.mistakes_generator("rhythm", np.random.uniform(min_noise, max_noise),
                                       np.random.uniform(min_percentage, max_percentage))
        performance.mistakes_generator("duration", np.random.uniform(min_noise, max_noise),
                                       np.random.uniform(min_percentage, max_percentage), False)
        performance.mistakes_generator("velocity", np.random.uniform(min_noise, max_noise),
                                       np.random.uniform(min_percentage, max_percentage), False)
        performance.mistakes_generator("pitch", np.random.uniform(min_noise, max_noise),
                                       np.random.uniform(min_percentage, max_percentage), False)
        flawed_performances.append(performance)

    return flawed_performances, original_midi_data


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
        note = pretty_midi.Note(velocity=int(m[3]), pitch=m[2], start=m[0], end=m[1])
        piano.notes.append(note)
    performance.instruments.append(piano)
    if write_midi_file:
        performance.write(midfilename)
    else:
        return_performance = Performance_class.Performance(path=None, name="name", player_name="name",
                                                           original_path=None,
                                                           prettyMidiFile_performance=performance,
                                                           prettyMidiFile_original=original_midi_file)
        return return_performance


def test_algorithms(labeled_data_train, labeled_data_test, with_tempo, to_print=True):
    if with_tempo:
        x_train = labeled_data_train.drop(columns=['label'])
        y_train = labeled_data_train['label']

        x_test = labeled_data_test.drop(columns=['label'])
        y_test = labeled_data_test['label']
        if to_print:
            print("##############")
            print("With Tempo:")
            print("##############")
    else:
        x_train = labeled_data_train.drop(columns=['label', 'Tempo'])
        y_train = labeled_data_train['label']

        x_test = labeled_data_test.drop(columns=['label', 'Tempo'])
        y_test = labeled_data_test['label']
        if to_print:
            print('')
            print("##############")
            print("Without Tempo:")
            print("##############")

    ### random forest

    model_rf_gini = RandomForestClassifier(criterion='gini')
    model_rf_gini.fit(x_train, y_train)
    random_forest_gini_score = model_rf_gini.score(x_test, y_test)

    model_rf_entropy = RandomForestClassifier(criterion='entropy')
    model_rf_entropy.fit(x_train, y_train)
    random_forest_entropy_score = model_rf_entropy.score(x_test, y_test)

    ### logistic regression (classification)

    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(x_train, y_train)
    logistic_regression_score = model_lr.score(x_test, y_test)

    ### knn (classification)
    max_knn_score = 0
    for i in range(3, 10):
        model_knn = KNeighborsClassifier(n_neighbors=i)
        model_knn.fit(x_train, y_train)
        knn_score = model_knn.score(x_test, y_test)
        if to_print:
            print("KNN with k = " + str(i) + " Score: " + str(knn_score))
        if knn_score > max_knn_score:
            max_knn_score = knn_score

    model_mlp = MLPClassifier()
    model_mlp.fit(x_train, y_train)
    mlp_score = model_mlp.score(x_test, y_test)
    ### MLP (classification)

    if to_print:
        print("Random Forest (gini) Score: " + str(random_forest_gini_score))
        print("Random Forest (entropy) Score: " + str(random_forest_entropy_score))
        print("Logistic Regression Score: " + str(logistic_regression_score))
        print("Multi-layer Perceptron with Neural Networks score: " + str(mlp_score))

    return random_forest_gini_score, random_forest_entropy_score, logistic_regression_score, max_knn_score, mlp_score
