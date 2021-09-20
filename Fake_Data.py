import Performance_class
import Song_Class
import numpy as np
import os
import shutil
from pathlib import Path
import auxiliary


def generate_random_mistakes_data(folder, n, create_midi_files, max_noise, max_percentage, min_noise=0, min_percentage=0.5):
    basepath = folder + '/'
    all_data = []
    if create_midi_files:
        fake_data_path = folder + ' - fake data/'
        Path(fake_data_path).mkdir(exist_ok=True)
    with os.scandir(basepath) as songs:
        for song in songs:
            song_name = song.name.split(".")[0]
            if song.is_file() and song.name != '.DS_Store':
                song_instance = Song_Class.Song(song_name)
                flawed_performances, original_midi_data = create_random_mistakes(basepath + song.name, song_name, n,
                                                                                 min_noise=min_noise, max_noise=max_noise,
                                                                                 min_percentage=min_percentage, max_percentage=max_percentage)
                if create_midi_files:
                    Path(fake_data_path + song_name).mkdir(exist_ok=True)
                    shutil.copy(basepath + song.name, fake_data_path + song_name)
                    Path(fake_data_path + song_name + '/fake performances/').mkdir(exist_ok=True)
                for i, data in enumerate(flawed_performances):
                    if create_midi_files:
                        path = fake_data_path + song_name + '/fake performances/' + song_name + str(i) + ".mid"
                        auxiliary.np2mid(data.midi_df, path, original_midi_file=None, write_midi_file=True)
                    else:
                        fake_data_performance = auxiliary.np2mid(data.midi_df, song_name, original_midi_data[i],
                                                                 write_midi_file=False)
                        song_instance.fake_performances.append(fake_data_performance)
            all_data.append(song_instance)
    return all_data


def create_random_mistakes(path, name, n, max_noise, max_percentage, min_noise=0, min_percentage=0.5):
    flawed_performances = []
    original_midi_data = []
    for i in range(n):
        performance = Performance_class.Performance(path, name, name + " random mistakes: " + str(i), path)
        original_midi_data.append(performance.midi_data_original)
        mistakes_generator(performance, "rhythm", np.random.uniform(min_noise, max_noise),
                           np.random.uniform(min_percentage, max_percentage))
        mistakes_generator(performance, "duration", np.random.uniform(min_noise, max_noise),
                           np.random.uniform(min_percentage, max_percentage), False)
        mistakes_generator(performance, "velocity", np.random.uniform(min_noise, max_noise),
                           np.random.uniform(min_percentage, max_percentage), False)
        mistakes_generator(performance, "pitch", np.random.uniform(min_noise / 2, max_noise / 4),
                           np.random.uniform(min_percentage, max_percentage), False)
        flawed_performances.append(performance)

    return flawed_performances, original_midi_data


def mistakes_generator(performance, feature, noise=0.75, percentage=1, original=True):
    if original:
        new_midi_df = np.copy(performance.midi_df_original)
        reference = performance.midi_df_original
    else:
        new_midi_df = np.copy(performance.midi_df)
        reference = performance.midi_df
    if noise > 1 or noise < 0 or percentage < 0 or percentage > 1:
        print("Input values should be between 0 and 1")

    if feature == "rhythm":
        accumulation_mistake = 0
        for i, note in enumerate(new_midi_df):
            if np.random.rand() < percentage and i > 0:
                accumulation_mistake += noise * (note[0] - reference[i - 1][0])
            note[0] += accumulation_mistake
            note[1] += accumulation_mistake
    if feature == "duration":
        for i, note in enumerate(new_midi_df):
            if np.random.rand() < percentage:
                if i + 1 == new_midi_df.shape[0] or note[1] + (note[1] - note[0]) * noise < reference[i + 1][0]:
                    note[1] += (note[1] - note[0]) * noise
                else:
                    note[1] = reference[i + 1][0] - 0.005
    if feature == "velocity":
        for note in new_midi_df:
            if np.random.rand() < percentage:
                new_value = note[3] * (1 + noise / 2)
                if 0 <= new_value <= 127:
                    note[3] = new_value
                else:
                    new_value = note[3] * (1 - noise / 2)
                    if 0 <= new_value <= 127:
                        note[3] = new_value
    if feature == "pitch":
        for note in new_midi_df:
            if np.random.rand() < percentage / 6:
                note[2] = 1 + note[2]
    performance.midi_df = new_midi_df
