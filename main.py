# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import Performance_class

test = Performance_class.Performance('musicsamples/midi_test_2021_05_14_18_30_58_Jason_BnuGesher.mid',
                                     'HaAviv', 'Jason', 'musicsamples/85c_BnuGesher.midi')
# HaAviv.visualise()
print(test.baseline_grader(sigma=5))
