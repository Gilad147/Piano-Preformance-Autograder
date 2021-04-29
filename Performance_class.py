import pretty_midi
import pandas as pd
import libfmp.c1
import baseline_graders as bl


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
                libfmp.c1.visualize_piano_roll(score, figsize=(8, 3), velocity_alpha=True)

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
