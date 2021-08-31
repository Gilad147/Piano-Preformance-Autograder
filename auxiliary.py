import Performance_class
import numpy as np
from midiutil.MidiFile import MIDIFile
import pretty_midi
import os
import shutil
from pathlib import Path


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
                                                                                    min_noise=0, max_noise=1,
                                                                                    min_percentage=0, max_percentage=1)
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
    return_performance = Performance_class.Performance(path=None, name="name", player_name="name", original_path=None,
                                                           prettyMidiFile_performance=performance,
                                                           prettyMidiFile_original=original_midi_file)
    return return_performance

