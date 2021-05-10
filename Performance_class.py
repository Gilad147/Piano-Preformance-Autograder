import pretty_midi
import pandas as pd
import libfmp.c1
import numpy as np
import baseline_graders as bl
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import colors


class Chord:
    def __init__(self, notes):
        self.notes = np.sort(notes, axis=0)
        self.root = np.min(notes[:, 0])
        self.is_single_note = (notes.shape[0] == 1)


class Performance:
    def __init__(self, path, name, player_name, original_path, is_original=0):
        self.midi_data = pretty_midi.PrettyMIDI(path)
        self.name = name
        self.is_original = is_original
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
        self.original = pd.DataFrame(midi_list_orig, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument']).to_numpy()
        """if is_original:
            originals_dict.append
            """

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

    def calculate_grade(self, original, grading_model="baseline"):
        """
        calculates the grades of the piece played in comparison to the original
        :param grading_model: which method will be used for grading, can be "baseline", ... , ...(ML graders)
        :param original: The original performance
        :return: a dictionary of 3 grades [Pitch,Timing,Dynamics(velocity)] between 0-100
        """

        dynamics = bl.dynamics_grader(original, self)
        pitch = bl.pitch_grader(original, self)
        timing = bl.timing_grader(original, self)
        # Implement the grades calculation
        return {"dynamics": dynamics, "pitch": pitch, "timing": timing}

    def get_chords_list(self):
        """

        :return: a list of the chords and single notes played in the performance
        """
        chords = []
        i = 0
        per_np = self.midi_df.to_numpy()
        while i < per_np.shape[0]:
            chord_index = 1

            while i + chord_index < per_np.shape[0] and abs(per_np[i][0] - per_np[i + chord_index][0]) < 0.0005 and \
                    abs(per_np[i][1] - per_np[i + chord_index][1]) < 0.05:
                chord_index += 1
            current_chord = per_np[i:i + chord_index][:, 2:4]
            chords.append(Chord(current_chord))
            i += chord_index
        return chords

    def find_missing_unnecessary_notes(self, original, timing_mistake=0.4):
        orig_np = original.midi_df.to_numpy()
        stud_np = self.midi_df.to_numpy()
        correct_orig = np.zeros(orig_np.shape[1] + 1)
        correct_stud = np.zeros(stud_np.shape[1] + 1)
        missing = np.zeros(orig_np.shape[0])
        unnecessary = np.zeros(stud_np.shape[0])
        i = 0
        while i < orig_np.shape[0]:
            j = 0
            start = orig_np[i][0]
            end = orig_np[i][1]
            while j < stud_np.shape[0] and end + timing_mistake > stud_np[j][0]:
                if abs(start - stud_np[j][0]) < timing_mistake and abs(end - stud_np[j][1]) < timing_mistake:
                    if missing[i] == 0:
                        cur_correct = np.append(orig_np[i], i)
                        correct_orig = np.vstack((correct_orig, cur_correct))
                        missing[i] = 0.5
                    if unnecessary[j] == 0:
                        cur_correct = np.append(stud_np[j], j)
                        unnecessary[j] = 0.5
                        correct_stud = np.vstack((correct_stud, cur_correct))
                j += 1
            i += 1

        return correct_orig, correct_stud, missing, unnecessary

    def grader(self):
        """
            :param timing_mistake:
            :param original: the original performance, a "Performance_class" object
            :return: pitch grade between 0-100, a list of the weights of original's each note, a list of the weights of stud's each note
            """
        orig = self.original
        stud_current = self.midi_df
        pitch_score = 0
        timing_score = 0
        velocity_score = 0

        for i in range(orig.shape[0]):
            current_note_orig = orig[i]
            start = current_note_orig[0] - 1
            if i != orig.shape[0] - 1:
                end = max(current_note_orig[1], orig[i+1][0])
            else:
                end = current_note_orig[1]
            pitch_orig = current_note_orig[2]
            j_start, j_end = self.find_j_start_and_end(start, end, stud_current)

            if j_start is not None and j_end is not None:
                while j_start <= j_end:
                    current_note_stud = stud_current[j_start]
                    if np.any(current_note_stud): #if current_note_stud is not all zeros (if we didn't count this note yet)
                        if current_note_stud[2] == pitch_orig:
                            stud_current[j_start] = np.zeros(stud_current.shape[1]) # if there is a match, "delete" the note by making the row all zeros
                            timing_score += self.timing_velocity_grader(i, j_start, True) # i = index of orig, j_start = index of stud
                            velocity_score += self.timing_velocity_grader(i, j_start, False) # i = index of orig, j_start = index of stud
                            pitch_score += 1
                            break
                    j_start += 1

        pitch_grade = pitch_score / orig.shape[0] # note missing issue - if a note sohuld have been played and the user missed it and didn't play any other note.
        too_many_notes_grade = self.calculate_too_many_notes_grade(stud_current) #too many notes issue - if a note should have been played and the user played other notes.
        timing_grade = self.score_to_grade(timing_score)
        velocity_grade = self.score_to_grade(velocity_score)
        timing_grade = 1 - timing_score / orig.shape[0]
        velocity_grade = 1 - velocity_score / orig.shape[0]
        return pitch_grade, too_many_notes_grade, timing_grade, velocity_grade


    def find_j_start_and_end(self, start, end, stud_current):
        j = 0
        while j < stud_current.shape[0]:
            if np.any(stud_current[j]): # if current_note_stud is not all zeros (if we didn't count this note yet)
                if stud_current[j][0] >= start:
                    j_start = j
                    while j < stud_current.shape[0]:
                        if np.any(stud_current[j]):
                            if stud_current[j][0] >= end:
                                return j_start, j
                        j += 1
                    return j_start, j-1
            j += 1
        return None, None

    def calculate_too_many_notes_grade(self, stud_new):
        too_many_notes_score = 0

        for i in range(stud_new.shape[0]):
            i_all_zero = not np.any(stud_new[i])
            if not i_all_zero:
                too_many_notes_score += 1

        return too_many_notes_score / stud_new.shape[0]

    def score_to_grade(self, score):
        return 0

    def timing_velocity_grader(self, i, j,
                               timing):  # timing = True for timing, timing = False for velocity. i = index of orig, j = index of stud
        orig = self.original
        stud_current = self.midi_df
        stud_note = stud_current[j]
        orig_note = orig[i]
        if timing:
            orig_duration = orig_note[1] - orig_note[0]
            stud_duration = stud_note[1] - stud_note[0]
            distance = np.abs(stud_note[0] - orig_note[0])
            diff_duration = np.abs(orig_duration - stud_duration)
            timing_grade = self.deviation_of_note(distance, sigma=2)
            duration_grade = self.deviation_of_note(diff_duration, sigma=2)
            return (2 / 3) * timing_grade + (1 / 3) * duration_grade
        else:
            orig_velocity = orig_note[3]
            stud_velocity = stud_note[3]
            diff_velocity = np.abs(orig_velocity - stud_velocity)
            velocity_grade = self.deviation_of_note(diff_velocity, sigma=70)
            return velocity_grade

    def deviation_of_note(self, distance, sigma=2):
        if sigma == 2:
            gausian = lambda x: np.exp(-0.5 * (x / sigma) ** 2)
        else:
            gausian = lambda x: np.exp(-0.5 * (np.max([(x - 8), 1]) / 70) ** 2)
        return 1 - np.round(gausian(distance), 3)

    def dynamics_grader(original, student):
        """

            :param original: the original performance, a "Performance_class" object
            :param student: the student's performance, a "Performance_class" object
            :return: dynamics grade between 0-100
            """
        grade = 0
        # TODO Implement the grades calculation
        return grade

    def timing_grader(original, student):
        """

        :param original: the original performance, a "Performance_class" object
            :param student: the student's performance, a "Performance_class" object
        :return: timing grade between 0-100
        """
        grade = 0
        # TODO Implement the grades calculation
        return grade
