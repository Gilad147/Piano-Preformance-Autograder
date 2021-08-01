import pretty_midi
import pandas as pd
import libfmp.c1
import numpy as np
from matplotlib import pyplot as plt
from difflib import SequenceMatcher
import sklearn as skl


class Chord:
    def __init__(self, notes):
        self.notes = np.sort(notes, axis=0)
        self.root = np.min(notes[:, 0])
        self.is_single_note = (notes.shape[0] == 1)


class Performance:
    def __init__(self, path, name, player_name, original_path):
        self.midi_data = pretty_midi.PrettyMIDI(path)
        self.name = name
        self.player_name = player_name

        midi_list = []
        for instrument in self.midi_data.instruments:
            for note in instrument.notes:
                start = note.start
                end = note.end
                pitch = note.pitch
                velocity = note.velocity
                midi_list.append([start, end, pitch, velocity, instrument.name])

        midi_list = sorted(midi_list, key=lambda x: (x[0], x[2]))
        self.midi_df = pd.DataFrame(midi_list, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument']).to_numpy()

        midi_data_orig = pretty_midi.PrettyMIDI(original_path)
        midi_list_orig = []
        for instrument in midi_data_orig.instruments:
            for note in instrument.notes:
                start = note.start
                end = note.end
                pitch = note.pitch
                velocity = note.velocity
                midi_list_orig.append([start, end, pitch, velocity, instrument.name])

        midi_list_orig = sorted(midi_list_orig, key=lambda x: (x[0], x[2]))
        self.original = pd.DataFrame(midi_list_orig,
                                     columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument']).to_numpy()

    def visualise(self):
        score = []
        for instrument in self.midi_data.instruments:
            for note in instrument.notes:
                start = note.start
                duration = note.end - start
                pitch = note.pitch
                velocity = note.velocity / 128.
                score.append([start, duration, pitch, velocity, instrument.name])
        fig, ax = libfmp.c1.visualize_piano_roll(score, figsize=(8, 3), velocity_alpha=True)
        plt.show()

    def get_chords_list(self):
        """

        :return: a list of the chords and single notes played in the performance
        """
        chords = []
        i = 0
        per_np = self.midi_df
        while i < per_np.shape[0]:
            chord_index = 1

            while i + chord_index < per_np.shape[0] and abs(per_np[i][0] - per_np[i + chord_index][0]) < 0.0005 and \
                    abs(per_np[i][1] - per_np[i + chord_index][1]) < 0.05:
                chord_index += 1
            current_chord = per_np[i:i + chord_index][:, 2:4]
            chords.append(Chord(current_chord))
            i += chord_index
        return chords

    def mistakes_generator(self, feature, noise=0.5, percentage=0.25):
        new_midi_df = self.original.__deepcopy__()
        if feature == "rhythm":
            accumulation_mistake = 0
            for i, note in enumerate(new_midi_df):
                if np.random.rand() < percentage and i > 0:
                    accumulation_mistake += noise * (note[0] - new_midi_df[i - 1][0])
                note[0] += accumulation_mistake
        if feature == "duration":
            for note in new_midi_df:
                if np.random.rand() < percentage:
                    note[1] += (note[1] - note[0]) * noise
        if feature == "velocity":
            for note in new_midi_df:
                if np.random.rand() < percentage:
                    note[4] *= noise
        return new_midi_df

    def baseline_grader(self, sigma=20):
        """

        :param sigma: sigma for error functions
        :return: Dictionary with performance grades
        """
        orig = self.original
        stud = self.midi_df
        tempo_score = 0
        a_d_score = 0
        matching_notes = 0
        orig_pitch_list = orig[:, 2]
        stud_pitch_list = stud[:, 2]
        # loop for checking blocks replayed
        for i in range(3):
            # returns list of triples describing matching subsequences
            matcher = SequenceMatcher(a=orig_pitch_list, b=stud_pitch_list)
            blocks = matcher.get_matching_blocks()
            if len(blocks) == 1:
                break
            scores = self.blocks_grader(blocks, sigma)
            if i == 0:
                played_notes = scores[2]
                blocks_count = len(blocks)
            tempo_score += scores[0] * (1 - i * 0.25)
            a_d_score += scores[1] * (1 - i * 0.25)
            matching_notes += scores[2]
            # if it is not the same song or the performance is messy don't iterate further
            if played_notes / orig.shape[0] < 0.5:
                break
        # pitch grade is effected by number of initial blocks, missing and unnecessary notes
        pitch_grade = played_notes / (orig.shape[0] * 2) + matching_notes / (stud.shape[0] * 2) - (
                blocks_count - 2) / orig.shape[0]
        pitch_grade = max(0, pitch_grade)
        # to avoid "over fitting" of the blocks, the grade is normalized with respect to pitch grade
        tempo_grade = (1 - tempo_score / matching_notes) * np.sqrt(pitch_grade)
        a_d_grade = (1 - a_d_score / matching_notes) * np.sqrt(pitch_grade)
        return {"pitch grade": pitch_grade, "tempo grade": tempo_grade, "articulation and dynamics grade": a_d_grade}

    def blocks_grader(self, blocks, sigma):
        """

        :param blocks: a list of triples describing student's matching sub-sequences played
        :param sigma: sigma for scoring functions
        :return: tempo grade, a_d_grade - articulation and dynamics grade, number of notes from the blocks that the student played.
        """
        orig = self.original
        stud = self.midi_df
        tempo_score = 0
        a_d_score = 0
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
            for i in range(block[2]):
                # testing the block's grades of timing and velocity
                cur_orig_note = orig[orig_start + i]
                cur_stud_note = stud[stud_start + i]
                # calculate relative timing for each note in block
                cur_orig_note[0] = cur_orig_note[0] - orig_set_time
                cur_orig_note[1] = cur_orig_note[1] - orig_set_time
                cur_stud_note[0] = cur_stud_note[0] - stud_set_time
                cur_stud_note[1] = cur_stud_note[1] - stud_set_time
                # ignore note in further analysis
                cur_stud_note[2] = 0
                # calculate grades for difference in notes
                tempo_note_score, a_d_note_score = self.timing_velocity_grader(cur_orig_note, cur_stud_note,
                                                                               sigma=sigma)
                tempo_score += tempo_note_score
                a_d_score += a_d_note_score
                # reset timing in original performance
                cur_orig_note[0] = cur_orig_note[0] + orig_set_time
                cur_orig_note[1] = cur_orig_note[1] + orig_set_time

        return tempo_score, a_d_score, matching_notes

    def get_features(self):
        orig = self.original
        stud = self.midi_df
        orig_pitch_list = orig[:, 2]
        stud_pitch_list = stud[:, 2]
        matcher = SequenceMatcher(a=orig_pitch_list, b=stud_pitch_list)
        blocks = matcher.get_matching_blocks()
        rhythm_diff, velocity_diff, duration_diff, matching_notes = self.supervised_blocks_diff(blocks)
        rhythm_feature = 1 - sum(rhythm_diff) / matching_notes
        velocity_feature = 1 - sum(velocity_diff) / matching_notes
        duration_feature = 1 - sum(duration_diff) / matching_notes
        pitch_feature = matching_notes / len(orig_pitch_list)
        return rhythm_feature, velocity_feature, duration_feature, pitch_feature

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
                cur_orig_note = orig[i]
                cur_stud_note = stud[i]

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
                rhythm_diff.append(np.abs(orig_rhythm - stud_rhythm) / (orig_rhythm + 0.0001))
                velocity_diff.append(np.abs(cur_orig_note[3] - cur_stud_note[3]) / cur_orig_note[3])
                orig_duration = cur_orig_note[1] - cur_orig_note[0]
                stud_duration = cur_stud_note[1] - cur_stud_note[0]
                duration_diff.append(np.abs(orig_duration - stud_duration) / orig_duration)
                # reset timing in original performance
                cur_orig_note[0] = cur_orig_note[0] + orig_set_time
                cur_orig_note[1] = cur_orig_note[1] + orig_set_time
        return rhythm_diff, velocity_diff, duration_diff, matching_notes

    def timing_velocity_grader(self, orig_note, stud_note, sigma=10):
        """

        :param orig_note:
        :param stud_note:
        :param sigma:
        :return: grades for differences between notes with the same pitch
        """

        orig_duration = orig_note[1] - orig_note[0]
        stud_duration = stud_note[1] - stud_note[0]
        distance = np.abs(stud_note[0] - orig_note[0])
        diff_duration = np.abs(orig_duration - stud_duration)
        timing_grade = self.deviation_of_note(distance, sigma=sigma // 2)
        duration_grade = self.deviation_of_note(diff_duration, sigma=sigma // 5)
        orig_velocity = orig_note[3]
        stud_velocity = stud_note[3]
        diff_velocity = np.abs(orig_velocity - stud_velocity)
        velocity_grade = self.deviation_of_note(diff_velocity, sigma=sigma, grade="velocity")
        return timing_grade, (2 / 3) * velocity_grade + (1 / 3) * duration_grade

    def deviation_of_note(self, distance, sigma=5, grade="timing"):
        if grade == "timing":
            gausian = lambda x: np.exp(-0.5 * (x / sigma) ** 2)
        else:
            gausian = lambda x: np.exp(-0.5 * (np.max([(x - 5), 1]) / sigma) ** 2)
        return 1 - np.round(gausian(distance), 3)
