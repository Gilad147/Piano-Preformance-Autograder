import pretty_midi
import pandas as pd
import numpy as np
from difflib import SequenceMatcher


class Performance:
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
        self.tempo = self.midi_data.estimate_tempo()
        midi_list = []
        for instrument in self.midi_data.instruments:
            for note in instrument.notes:
                start = note.start
                end = note.end
                pitch = note.pitch
                velocity = note.velocity
                midi_list.append([start, end, pitch, velocity, instrument.name])

        self.midi_df = pd.DataFrame(midi_list,
                                    columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument']).to_numpy()
        self.midi_df = np.sort(self.midi_df, 0)

        self.orig_tempo = self.midi_data_original.estimate_tempo()
        midi_list_orig = []
        for instrument in self.midi_data_original.instruments:
            for note in instrument.notes:
                start = note.start
                end = note.end
                pitch = note.pitch
                velocity = note.velocity
                midi_list_orig.append([start, end, pitch, velocity, instrument.name])

        self.original = pd.DataFrame(midi_list_orig,
                                     columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument']).to_numpy()
        self.original = np.sort(self.original, 0)

        self.teachers_grades = []  # [[Teacher's Pitch score, Teacher's Tempo score, Teacher's Rhythm score,
        # Teacher's Articulation & Dynamics score, Teacher's next step]] (similar to the order in qualtrics)

        self.labels = []  # [Pitch, Tempo, Rhythm, Articulation & Dynamics, Next step]

    def predict_grades(self, technical_grades):
        return None

    def predict_reccomendation(self, technical_grades):
        return None

    def get_features(self):
        try:
            orig = self.original
            stud = self.midi_df
            orig_pitch_list = orig[:, 2]
            stud_pitch_list = stud[:, 2]
            matcher = SequenceMatcher(a=orig_pitch_list, b=stud_pitch_list)
            blocks = matcher.get_matching_blocks()
            rhythm_diff, velocity_diff, duration_diff, matching_notes = self.supervised_blocks_diff(blocks)
            if matching_notes == 0:
                return 0, 0, 0, 0, 0
            rhythm_feature = 1 - sum(rhythm_diff) / matching_notes
            velocity_feature = 1 - sum(velocity_diff) / matching_notes
            duration_feature = 1 - sum(duration_diff) / matching_notes
            pitch_feature = matcher.ratio()
            tempo_feature = 1 - (min(abs(self.orig_tempo - self.tempo) / self.orig_tempo, 1))
            return rhythm_feature, velocity_feature, duration_feature, pitch_feature, tempo_feature
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
        for block in blocks:
            # end of blocks list
            if block[2] == 0:
                break
            # ignore single notes
            if block[2] == 1:
                continue
            matching_notes += block[2]

            # match timing of the two matching parts
            orig_start = block[0]
            stud_start = block[1]
            orig_set_time = orig[orig_start, 0]
            stud_set_time = stud[stud_start, 0]
            for i in range(orig_start, orig_start + block[2]):
                # testing the block's grades of timing and velocity
                cur_orig_note = np.copy(orig[i])
                cur_stud_note = np.copy(stud[i])

                # ignore note in further analysis
                cur_stud_note[2] = 0
                # calculate grades for difference in notes
                if i > 0:
                    prev_orig = orig[i - 1]
                    prev_stud = stud[i - 1]

                    orig_rhythm = cur_orig_note[0] - prev_orig[0]
                    stud_rhythm = cur_stud_note[0] - prev_stud[0]
                else:
                    orig_rhythm = orig_set_time
                    stud_rhythm = stud_set_time
                rhythm_diff.append(np.abs(orig_rhythm - stud_rhythm) / (orig_rhythm + 0.00001))
                velocity_diff.append(np.abs(cur_orig_note[3] - cur_stud_note[3]) / cur_orig_note[3])
                orig_duration = cur_orig_note[1] - cur_orig_note[0]
                stud_duration = cur_stud_note[1] - cur_stud_note[0]
                duration_diff.append(np.abs(orig_duration - stud_duration) / orig_duration)

        return rhythm_diff, velocity_diff, duration_diff, matching_notes

    def give_labels(self, majority_or_avg):  # majority_or_avg == true --> majority ; majority_or_avg == false --> avg
        pitch_scores = [teacher[0] for teacher in self.teachers_grades]
        tempo_scores = [teacher[1] for teacher in self.teachers_grades]
        rhythm_scores = [teacher[2] for teacher in self.teachers_grades]
        a_d_scores = [teacher[3] for teacher in self.teachers_grades]
        next_step = [teacher[5] for teacher in self.teachers_grades]

        if majority_or_avg:
            labels = [max(set(pitch_scores), key=pitch_scores.count),
                      max(set(tempo_scores), key=tempo_scores.count),
                      max(set(rhythm_scores), key=rhythm_scores.count),
                      max(set(a_d_scores), key=a_d_scores.count),
                      max(set(next_step), key=next_step.count)]
        else:
            labels = [str(round((sum(list(map(int, pitch_scores))) / len(pitch_scores)))),
                      str(round((sum(list(map(int, tempo_scores))) / len(tempo_scores)))),
                      str(round((sum(list(map(int, rhythm_scores))) / len(rhythm_scores)))),
                      str(round((sum(list(map(int, a_d_scores))) / len(a_d_scores)))),
                      str(round((sum(list(map(int, next_step))) / len(next_step))))]

        self.labels = labels
