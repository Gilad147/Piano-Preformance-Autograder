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
    def __init__(self, path, name, player_name, is_original=0):
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
        self.midi_df = pd.DataFrame(midi_list, columns=['Start', 'End', 'Pitch', 'Velocity', 'Instrument'])
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

    def pitch_grader(self, original, timing_mistake=0.1):
        """
            :param timing_mistake:
            :param original: the original performance, a "Performance_class" object
            :return: pitch grade between 0-100, a list of the weights of original's each note, a list of the weights of stud's each note
            """

        correct_notes = 0
        close_notes = 0
        correct_orig, correct_stud, missing, unnecessary = self.find_missing_unnecessary_notes(original)
        i = 0
        while i < correct_orig.shape[0]:
            j = 0
            start = correct_orig[i][0]
            end = correct_orig[i][1]
            pitch = correct_orig[i][2]
            close = False
            while j < correct_stud.shape[0] and end + timing_mistake > correct_stud[j][0]:
                if abs(start - correct_stud[j][0]) < timing_mistake and abs(end - correct_stud[j][1]) < timing_mistake:
                    if pitch == correct_stud[j][2]:
                        correct_notes += 1
                        unnecessary[int(correct_stud[j][5])] = 1
                        if close:
                            close_notes -= 1
                        break
                    if abs(pitch - correct_stud[j][2]) == 1:
                        if not close:
                            close_notes += 1
                            close = True
                if close:
                    unnecessary[int(correct_stud[j][5])] = 0.75
                j += 1
            i += 1

        return 100 * (correct_notes / correct_orig.shape[0]) + 50 * (
                    close_notes / correct_orig.shape[0]), missing, unnecessary

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
