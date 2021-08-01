# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import Performance_class

test = Performance_class.Performance('musicsamples/midi_test_2021_05_14_18_44_33_Peleg_HaKovaSheli.mid',
                                     'HaAviv', 'Jason', 'musicsamples/74a_HaKovaSheli.midi')
# HaAviv.visualise()
print(test.get_features())
