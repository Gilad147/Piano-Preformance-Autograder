import pandas as pd


class Song:
    def __init__(self, name, level=None):
        self.name = name
        self.level = level
        self.performances = pd.DataFrame(columns=['Rhythm', 'Dynamics', 'Articulation', 'Pitch', 'Tempo', # every row is a different performance
                                                  "Teacher's Pitch", "Teacher's Rhythm", "Teacher's Tempo",
                                                  "Teacher's Articulation & Dynamics", 'label'])