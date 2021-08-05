# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import time

import Performance_class
import auxiliary

test_data = auxiliary.create_random_mistakes("musicsamples/91a_GinaLi.midi", "GinaLi", 100, min_noise=0,
                                             max_noise=0.5, min_percentage=0.1, max_percentage=0.5)
for i, data in enumerate(test_data):
    path = "musicsamples/GinaLi/test" + str(i) + ".mid"
    auxiliary.np2mid(data.midi_df, path)
    """
    try:
        feat_test = Performance_class.Performance(path, "i", "com", "musicsamples/91a_GinaLi.midi")
        print(feat_test.get_features())
    except:
        print("error in sample" + str(i+1))
    """
