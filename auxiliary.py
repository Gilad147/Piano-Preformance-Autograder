import Fake_Data
import pretty_midi

import Performance_class


def change_midi_file_tempo(original_path, new_path, percentage=0.10):
    performance = Performance_class.Performance(original_path, " ", " ",
                                                original_path)
    percentage = -percentage
    if percentage > 0:
        Fake_Data.mistakes_generator(performance, "rhythm", noise=percentage)
        Fake_Data.mistakes_generator(performance, "duration", noise=percentage, original=False)
    else:
        Fake_Data.mistakes_generator(performance, "duration", noise=percentage)
        Fake_Data.mistakes_generator(performance, "rhythm", noise=percentage, original=False)
    np2mid(performance.midi_df, new_path, None, True)
    return new_path


def np2mid(np_performance, midfilename, original_midi_file, write_midi_file):
    """
    Converts an numpy array  to a .mid file

    @param np_performance: np array with Midi values
    @param midfilename: full path to the mid output file
    @return: None
    """

    performance = pretty_midi.PrettyMIDI()

    piano = pretty_midi.Instrument(program=4)
    # Iterate over note names, which will be converted to note number later
    for m in np_performance:
        note = pretty_midi.Note(velocity=int(m[3]), pitch=int(m[2]), start=m[0], end=m[1])
        piano.notes.append(note)
    performance.instruments.append(piano)
    if write_midi_file:
        performance.write(midfilename)
    else:
        return_performance = Performance_class.Performance(path=None, name=midfilename, player_name="np2mid",
                                                           original_path=None,
                                                           prettyMidiFile_performance=performance,
                                                           prettyMidiFile_original=original_midi_file)
        return return_performance
