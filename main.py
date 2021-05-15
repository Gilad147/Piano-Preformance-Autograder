# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import Performance_class

littleY = Performance_class.Performance('LittleY_mixed.aif.MID', "littleY", "Y", 'LittleYon_perfect.aif.MID')
print(littleY.baseline_grader())
littleY_timing = Performance_class.Performance('LittleYon_timing.aif.MID', "littleY", "Y", 'LittleYon_perfect.aif.MID')
print(littleY_timing.baseline_grader())
littleY_missing_notes = Performance_class.Performance('LittleY_missingnotes.aif.MID', "littleY", "Y",
                                                 'LittleYon_perfect.aif.MID')
print(littleY_missing_notes.baseline_grader())
littleY_double_notes = Performance_class.Performance('LittleY_double_notes_3.aif.MID', "littleY", "Y",
                                                 'LittleYon_perfect.aif.MID')
print(littleY_double_notes.baseline_grader())
littleY_wrong_pitch = Performance_class.Performance('LittleY_wrong_pitch.aif.MID', "littleY", "Y",
                                                 'LittleYon_perfect.aif.MID')
print(littleY_wrong_pitch.baseline_grader())
littleY_bad = Performance_class.Performance('LittleYon_bad.aif.MID', "littleY", "Y",
                                                 'LittleYon_perfect.aif.MID')
print(littleY_bad.baseline_grader())