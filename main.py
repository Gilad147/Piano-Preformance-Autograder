import time

import Performance_class
import auxiliary
import Automated_teacher

# test_data = auxiliary.create_random_mistakes("musicsamples/91a_GinaLi.midi", "GinaLi", 100, min_noise=0,
#                                              max_noise=1, min_percentage=0, max_percentage=1)
#
# for i, data in enumerate(test_data):
#     path = "fake music samples/91a_GinaLi/91a_GinaLi Performances/GinaLi" + str(i) + ".mid"
#
#     auxiliary.np2mid(data.midi_df, path)
#
# Automated_teacher.fake_teachers_algorithm('fake music samples')
auxiliary.change_midi_file_tempo("songs/original songs - fake data/HaKova Sheli/HaKova Sheli.midi","songs/original songs - fake data/HaKova Sheli/slowdown_33.midi",-0.33333)

# without creating MIDI files
generated_data = auxiliary.generate_random_mistakes_data('original songs', 4, False)
Automated_teacher.fake_teachers_algorithm(False, performances_data=generated_data, number_of_teachers=10, train_ratio=0.7, majority_or_avg=True)


# creating MIDI files
# generated_data = auxiliary.generate_random_mistakes_data('original songs', 1000, True)
# Automated_teacher.fake_teachers_algorithm(from_midi_files_or_not=True, folder='original songs',
#                                           number_of_teachers=10, train_ratio=0.3)


#print(auxiliary.generate_random_mistakes_data('original songs', 2, True))