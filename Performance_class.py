import pretty_midi
import pandas as pd
import libfmp.c1
import numpy as np
from matplotlib import pyplot as plt
from difflib import SequenceMatcher


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
        # to avoid "overfitting" of the blocks, the grade is normalized with respect to pitch grade
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

    def guitar_hero_grader(self, timing_window=1):
        """
            :param timing_window:
            :return: grades between 0-100
            """
        orig = self.original
        stud_current = self.midi_df
        pitch_score = 0
        timing_score = 0
        velocity_score = 0
        played_notes = 0
        for i in range(orig.shape[0]):
            current_note_orig = orig[i]
            start = current_note_orig[0] - timing_window
            if i != orig.shape[0] - 1:
                end = max(current_note_orig[1], orig[i + 1][0])
            else:
                end = current_note_orig[1]
            pitch_orig = current_note_orig[2]
            j_start, j_end = self.find_j_start_and_end(start, end, stud_current, timing_window)

            note_played = False
            if j_start is not None and j_end is not None:
                while j_start <= j_end:
                    current_note_stud = stud_current[j_start]
                    if np.any(current_note_stud):
                        # if current_note_stud is not all zeros (if we didn't count this note yet)
                        if current_note_stud[2] == pitch_orig:
                            timing_score += self.timing_velocity_grader(current_note_orig, current_note_stud, True,
                                                                        sigma=10)
                            # i = index of orig, j_start = index of stud
                            velocity_score += self.timing_velocity_grader(current_note_orig, current_note_stud, False,
                                                                          sigma=5)
                            # i = index of orig, j_start = index of stud
                            pitch_score += 1
                            note_played = True
                            stud_current[j_start] = np.zeros(stud_current.shape[1])
                            # if there is a match, "delete" the note by making the row all zeros
                            break
                        if abs(current_note_stud[2] - pitch_orig) < 3:
                            note_played = True
                    j_start += 1
            if note_played:
                played_notes += 1
        too_many_notes_grade = min(
            self.calculate_too_many_notes_grade(stud_current) + (played_notes - pitch_score) / stud_current.shape[0], 1)
        # too many notes issue - if a note should have been played and the user played other notes.
        pitch_grade = pitch_score / played_notes
        # note missing issue - if a note should have been played and the user missed it and didn't play any other note.
        timing_grade = 1 - timing_score / orig.shape[0]
        velocity_grade = 1 - velocity_score / played_notes
        return pitch_grade, too_many_notes_grade, timing_grade, velocity_grade

    def find_j_start_and_end(self, start, end, stud_current, timing_window):
        j = 0
        while j < stud_current.shape[0]:
            if np.any(stud_current[j]):  # if current_note_stud is not all zeros (if we didn't count this note yet)
                if start <= stud_current[j][0] <= start + timing_window * 6:
                    j_start = j
                    while j < stud_current.shape[0]:
                        if np.any(stud_current[j]):
                            if stud_current[j][0] >= end:
                                return j_start, j
                        j += 1
                    return j_start, j - 1
            j += 1
        return None, None

    def calculate_too_many_notes_grade(self, stud_new):
        too_many_notes_score = 0

        for i in range(stud_new.shape[0]):
            i_all_zero = not np.any(stud_new[i])
            if not i_all_zero:
                too_many_notes_score += 1

        return 1 - too_many_notes_score / stud_new.shape[0]

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
