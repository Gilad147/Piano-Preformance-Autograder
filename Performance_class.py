import pretty_midi
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from data_functions import load_models, predict_from_models

def process_midi_to_numpy(midi_data: pretty_midi.PrettyMIDI):
    midi_list = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            start = note.start
            end = note.end
            pitch = note.pitch
            velocity = note.velocity
            midi_list.append([start, end, pitch, velocity, instrument.name])
    midi_df = pd.DataFrame(midi_list,
                           columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
    return midi_df.to_numpy()

class Performance:
    """
    note that tempo is not being calculated for songs with <20 different note start times
    """

    def __init__(self, path, name, player_name, original_path, prettyMidiFile_performance=None,
                 prettyMidiFile_original=None):
        self.name = name
        self.player_name = player_name

        if prettyMidiFile_performance is None and prettyMidiFile_original is None:
            self.midi_data = pretty_midi.PrettyMIDI(path)
            self.midi_data_original = pretty_midi.PrettyMIDI(original_path)
        else:
            self.midi_data = prettyMidiFile_performance
            self.midi_data_original = prettyMidiFile_original

        if len(self.midi_data.instruments) > 1:
            one_instrument = self.midi_data.instruments[0]
            for i in range(1, len(self.midi_data.instruments)):
                one_instrument.notes += self.midi_data.instruments[i].notes
            one_instrument.notes = list(set(one_instrument.notes))
            one_instrument.notes.sort(key=lambda x: x.start)
            self.midi_data.instruments = [one_instrument]

        self.midi_df = process_midi_to_numpy(self.midi_data)

        notes_set_for_tempo = set([x.start for x in self.midi_data.instruments[0].notes])
        if len(notes_set_for_tempo) < 20:
            self.tempo = -1
        else:
            self.tempo = self.midi_data.estimate_tempo()

        if len(self.midi_data_original.instruments) > 1:
            one_instrument_original = self.midi_data_original.instruments[0]
            for i in range(1, len(self.midi_data_original.instruments)):
                one_instrument_original.notes += self.midi_data_original.instruments[i].notes
            one_instrument_original.notes = list(set(one_instrument_original.notes))
            one_instrument_original.notes.sort(key=lambda x: x.start)
            self.midi_data_original.instruments = [one_instrument_original]

        self.original = process_midi_to_numpy(self.midi_data_original)

        notes_set_for_tempo_original = set([x.start for x in self.midi_data_original.instruments[0].notes])
        if len(notes_set_for_tempo_original) < 20:
            self.orig_tempo = -1
        else:
            self.orig_tempo = self.midi_data_original.estimate_tempo()

        self.teachers_grades = []  # [[Teacher's Pitch score, Teacher's Tempo score, Teacher's Rhythm score,
        # Teacher's Articulation & Dynamics score, Teacher's next step]] (similar to the order in qualtrics)

        self.labels = []  # [Pitch, Tempo, Rhythm, Articulation & Dynamics, Next step]


    def predict_grades(self, technical_grades):
        technical_grades = pd.DataFrame([technical_grades], columns=["Pitch", "Tempo", "Rhythm", "Articulation",
                                                                     "Dynamics"])

        ### Pitch
        x_pitch = pd.DataFrame(technical_grades["Pitch"])
        models_pitch = load_models("Pitch")
        pitch_prediction = str(predict_from_models(models_pitch, x_pitch))

        ### Tempo
        x_tempo = pd.DataFrame(technical_grades[["Pitch", "Tempo"]])
        models_tempo = load_models("Tempo")
        tempo_prediction = str(predict_from_models(models_tempo, x_tempo))

        ### Rhythm
        x_rhythm = pd.DataFrame(technical_grades["Rhythm"])
        models_rhythm = load_models("Rhythm")
        rhythm_prediction = str(predict_from_models(models_rhythm, x_rhythm))

        ### A&D
        x_a_d = pd.DataFrame(technical_grades[["Pitch", "Articulation", "Dynamics"]])
        models_a_d = load_models("Articulation & Dynamics")
        a_d_prediction = str(predict_from_models(models_a_d, x_a_d))

        ### Overall
        x_overall = pd.DataFrame(technical_grades[["Pitch", "Tempo", "Rhythm", "Articulation", "Dynamics"]])
        models_overall = load_models("Overall")
        overall_prediction = str(predict_from_models(models_overall, x_overall))

        return pitch_prediction, tempo_prediction, rhythm_prediction, a_d_prediction, overall_prediction

    def predict_reccomendation(self, technical_grades):
        technical_grades = pd.DataFrame([technical_grades], columns=["Pitch", "Tempo", "Rhythm", "Articulation",
                                                                   "Dynamics"])
        ### one_dim
        x_one_dim = pd.DataFrame(technical_grades[["Pitch", "Tempo", "Rhythm", 'Articulation', 'Dynamics']])
        models_one_dim = load_models("label_one_dim")
        one_dim_prediction = str(predict_from_models(models_one_dim, x_one_dim))

        return one_dim_prediction

    def get_features(self):
        try:

            orig = self.original
            stud = self.midi_df
            orig_pitch_list = orig[:, 2]
            stud_pitch_list = stud[:, 2]
            matcher = SequenceMatcher(a=orig_pitch_list, b=stud_pitch_list)
            blocks = matcher.get_matching_blocks()
            rhythm_diff, dynamics_diff, articulation_diff, matching_notes = self.supervised_blocks_diff(blocks)
            if matching_notes == 0:
                return 0, 0, 0, 0, 0

            rhythm_feature = 1 - (sum(rhythm_diff) / matching_notes)
            dynamics_feature = 1 - (sum(dynamics_diff) / matching_notes)
            articulation_feature = 1 - (sum(articulation_diff) / matching_notes)
            pitch_feature = matcher.ratio()

            if self.orig_tempo == -1 or self.tempo == -1:
                tempo_feature = float(1)
            else:
                tempo_feature = 1 - (min(abs(self.orig_tempo - self.tempo) / self.orig_tempo, 1))

            return pitch_feature, tempo_feature, rhythm_feature, articulation_feature, dynamics_feature
        except:
            return -1, 0, 0, 0, 0

    def mistakes_generator(self, feature, noise=0.75, percentage=1, original=True):
        if original:
            new_midi_df = np.copy(self.original)
            reference = self.original
        else:
            new_midi_df = np.copy(self.midi_df)
            reference = self.midi_df
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
                if np.random.rand() < percentage / 2:
                    note[2] = 1 + note[2]
        self.midi_df = new_midi_df

    def supervised_blocks_diff(self, blocks):
        """

        :param blocks:
        :return:
        """

        # technical features: tempo, velocity, note duration

        orig = self.original
        stud = self.midi_df
        rhythm_diff = []
        velocity_diff = []
        duration_diff = []
        matching_notes = 0
        for j, block in enumerate(blocks):
            # end of blocks list
            if block[2] == 0:
                break
            # ignore single notes
            if block[2] == 1:
                continue
            matching_notes += block[2]
            # match timing of the two matching parts
            orig_index = block[0]
            stud_index = block[1]
            orig_set_time = orig[orig_index, 0]
            stud_set_time = stud[stud_index, 0]
            # add rhythm differences between blocks
            if j != 0:
                orig_rhythm = orig_set_time - cur_orig_note[0]
                stud_rhythm = stud_set_time - cur_stud_note[0]
                rhythm_diff.append(np.abs(orig_rhythm - stud_rhythm) / orig_rhythm + 0.005)

            for i in range(block[2]):
                # testing the block's grades of timing and velocity
                cur_orig_note = np.copy(orig[orig_index])
                cur_stud_note = np.copy(stud[stud_index])

                # ignore note in further analysis
                cur_stud_note[2] = 0
                # calculate grades for difference in notes
                if i > 0:
                    prev_orig = orig[orig_index - 1]
                    prev_stud = stud[stud_index - 1]

                    orig_rhythm = cur_orig_note[0] - prev_orig[0]
                    stud_rhythm = cur_stud_note[0] - prev_stud[0]
                else:
                    orig_rhythm = 0
                    stud_rhythm = 0

                if orig_rhythm != 0:
                    rhythm_diff.append(np.abs(orig_rhythm - stud_rhythm) / orig_rhythm)

                velocity_diff.append(np.abs(cur_orig_note[3] - cur_stud_note[3]) / cur_orig_note[3])
                orig_duration = cur_orig_note[1] - cur_orig_note[0]
                stud_duration = cur_stud_note[1] - cur_stud_note[0]
                duration_diff.append(np.abs(orig_duration - stud_duration) / orig_duration)

                orig_index += 1
                stud_index += 1

        return rhythm_diff, velocity_diff, duration_diff, matching_notes

    def give_labels(self, majority_or_avg):  # majority_or_avg == true --> majority ; majority_or_avg == false --> avg
        pitch_scores = [teacher[0] for teacher in self.teachers_grades]
        tempo_scores = [teacher[1] for teacher in self.teachers_grades]
        rhythm_scores = [teacher[2] for teacher in self.teachers_grades]
        a_d_scores = [teacher[3] for teacher in self.teachers_grades]
        overall_scores = [teacher[4] for teacher in self.teachers_grades]
        next_step = [teacher[5] for teacher in self.teachers_grades]

        if majority_or_avg:
            labels = [max(set(pitch_scores), key=pitch_scores.count),
                      max(set(tempo_scores), key=tempo_scores.count),
                      max(set(rhythm_scores), key=rhythm_scores.count),
                      max(set(a_d_scores), key=a_d_scores.count),
                      max(set(overall_scores), key=overall_scores.count),
                      max(set(next_step), key=next_step.count)]
        else:
            labels = [(round((sum(list(map(int, pitch_scores))) / len(pitch_scores)))),
                      (round((sum(list(map(int, tempo_scores))) / len(tempo_scores)))),
                      (round((sum(list(map(int, rhythm_scores))) / len(rhythm_scores)))),
                      (round((sum(list(map(int, a_d_scores))) / len(a_d_scores)))),
                      (round((sum(list(map(int, overall_scores))) / len(a_d_scores)))),
                      (round((sum(list(map(int, next_step))) / len(next_step))))]

        self.labels = labels

