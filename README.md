# Piano-Preformance-Autograder
## Table of contents
* [General info](#general-info)
* [GUI Setup](#gui-setup)
* [Run the GUI](#run-the-gui)
* [Run experiments with the data](#run-experiments-with-the-data)
* [Fake data & Fake teachers](#fake-data-and-fake-teachers)

## General info
This project is designed to make automatic evaluations of piano performances recorded with a MIDI keyboard.
The evaluation is achieved using a combination of different Machine Learning algorithms.
Asside from getting a numerical grade for features of the performance, the player get a Next-Step improving recommendation.
	
## GUI Setup
The project contains a directory with 43 different musical pieces, grouped as Song Books. 
To add new musical pieces:
* Go to /project directory/songs/
* Choos a Song Book to add the song to. Create a new folder if you want to add a new Song Book.
* Add the following files for each song:
   *  <song_name>.ly
   *  <song_name>.midi
   *  <song_name>.png
* Data recorded while using the app will be saved in the directoy /students recordings.
	
## Run the GUI
To run the application, clone the project and open it using your local Python IDE.
* Install all required packages: 
   * Open Terminal/Command Line and go to the same directory where you saved the project's files (Alternatively - use the Terminal in your Python IDE, if this feature exists in your IDE).
   * Run the command:
   ```
   pip3 install -r requirements.txt
   ```
* Run the file "GUI.py".

## Run experiments with the data
<details>
<summary><b>Initializing a "Performance" class instance</b></summary>

	
All of the users' performances are being saved as instances of the class "Performance", that can be found in the file **"Performance_class.py"**.
To initialize an instance of this class you will have to pass the following parameters:
* ```path``` - the path of the MIDI file of the performance. If you would like to initialize an instance for a fake performance that do not have a MIDI file, pass an arbitrary string.
* ``` name ``` - the name of the song played.
* ```player_name``` - the name of the player. If you would like to initialize an instance for a fake performance that do not have a MIDI file, pass an arbitrary string.
* ```original_path``` - the path of the MIDI file of the "perfect" performance.
* ```prettyMidiFile_performance``` - optional. This parameter is for using fake data. When using real data, pass None (it's also the default value). If the performance do not have a MIDI file (meaning it's fake), pass the prettyMIDI instance of the performance.
* ```prettyMidiFile_original``` - optional. This parameter is for using fake data. When using real data, pass None (it's also the default value). If the performance do not have a MIDI file (meaning it's fake), pass the prettyMIDI instance of the "perfect" performance.	

After initializing, a Performance class' instance has few more fields:
* ```teacher's grades``` - a list that contains all of the teacher's grades. 
* ```labels``` - a list that will contain the final grades and next step reccomendation of the performance. To have the performance graded, the field "teachers_grades" must not be empty, and the function "give_labels" of the class "Performance_class" should be called.
* ```tempo``` - the performance's tempo.
* ```original_tempo``` - the "perfect" performance's tempo.
	
</details>

<details>
<summary><b>Initializing a "Song" class instance</b></summary>

	
All of the songs are being saved as instances of the class "Song", that can be found in the file **"Song_Class.py"**.
To initialize an instance of this class you will have to pass the following parameters:
* ```name``` - the name of the song.
* ``` level ``` - optional. The level of the song (the default value is None).
* ```player_name``` - the name of the player. If you would like to initialize an instance for a fake performance that do not have a MIDI file, pass an arbitrary string.
* ```original_path``` - the path of the MIDI file of the "perfect" performance.
* ```prettyMidiFile_performance``` - this parameter is for using fake data. When using real data, pass None (it's also the default value). If the performance do not have a MIDI file (meaning it's fake), pass the prettyMIDI instance of the performance.
* ```prettyMidiFile_original``` - this parameter is for using fake data. When using real data, pass None (it's also the default value). If the performance do not have a MIDI file (meaning it's fake), pass the prettyMIDI instance of the "perfect" performance.
	
After initializing, a Performance class' instance has few more fields:
* ```perfroamcnes``` - a list that will contain all of this song's performances. 
* ```fake_performances``` - a list that will contain all of this song's fake performances. 
	
</details>


## Fake data and Fake teachers

### Fake data
You can generate fake users' performances data based on existing songs. This algorithm inserts random noise to each MIDI feature.
Use the function **"generate_random_mistakes_data"** in the **"Fake_Data.py"** file, with the following parameters:
* ```folder``` - the path of the folder that contains the songs you would like to generate fake data based on. 
* ``` n ``` - number of fake performances generated out of **every song in the directory**. (For example - if n=3 and the directory contains 4 songs - 12 fake performances will be generated).
* ```create_midi_files``` - 'true' if you want to create a MIDI file for each fake performance you generate. 'false' otherwise (in this case you will have to use the return value of this function if you would like to use the fake generated data outside of this function).
* ```max_noise``` - a number between 0 to 1. Higher number will raise the magnitude ceiling for each change of the original song
* ```max_percentage``` - a number between 0 to 1.  Higher number will raise the ceiling of the percentage of the song notes to be cahnged 
* ```min_noise``` - a number between 0 to 1.  Higher number will raise the magnitude floor for each change of the original song
* ```min_percentage``` - a number between 0 to 1.  Higher number will raise the floor of the percentage of the song notes to be cahnged 

**return value of the function** - the functions returns a list of instances of the class "Song_Class". Each instance represents a song, and all its fake performances are in the field 'fake_performances'.

### Fake teachers
We created a fake teachers algorithm that can grade performances and give a next step reccomendation. 

<details>
<summary><b>Create fake teachers</b></summary>

	
Use the function **create_fake_teachers** in the **"Automated_teacher.py"** file, with the following parameters:
* ```number_of_teachers``` - number of different fake teachers to be created.

**return value of the function** - the function returns a list of "Teacher" class instances, representing the different teachers created.
	
</details>

<details>
<summary><b>Grade performances using the fake teacherss</b></summary>

	
Use the function **fake_teachers_feedback** in the **"Automated_teacher.py"** file, with the following parameters:
* ```performance``` - the performance that you want to be graded by the fake teachers, as "Performance" class instance.
* ```teachers``` - a list containing "Teacher" class' instances for each fake teacher (the output of the function "create_fake_teahcers", and can be found in the field ).
* ```pitch_tech_score``` - the pitch technical score of the performance (computed using the function "get_features" of the class "Performance").
* ```tempo_tech_score``` - the tempo technical score of the performance (computed using the function "get_features" of the class "Performance").
* ```rhythm_tech_score``` - the rhythm technical score of the performance (computed using the function "get_features" of the class "Performance").
* ```articulation_tech_score``` - the articulation technical score of the performance (computed using the function "get_features" of the class "Performance").
* ```dynamics_tech_score``` - the dynamics technical score of the performance (computed using the function "get_features" of the class "Performance").

**return value of the function** - the function do not return anything. It adds each fake teacher's grades to the field "teachers_grades" of the performance. 

</details>

<details>
<summary><b>Adjust the fake teachers' grading algorithms</b></summary> 
	
All of the algorithms that are being used to grade a performance are thresholds-based algorithms. Each threshold is in actual fact a fixed value + randomly selected unique teacher's value (can be positive/negative).
You can adjust the fake teachers' grading & next step algorithms, by adjusting one or more of the following parameters:
* **Unique values of each teacher** -  you can set the range in which the unique value is being randomly selected from, for each feature seperatly. The ranges can be found in the function "create_fake_teachers" in the "Automated_teacher.py" file:
  * The variable "featureName_unique_next_step" - for the next step algorithm.
  * The variable "featureName_unique_score" - for the grades algorithm.
* **Different thresholds** - you can change the thresholds in one or more algorithms. The algorithms can be found in the following functions:
  * The function "give_next_step_recco" in the class "Teachers" for the next step reccomendations.
  * The function "give_scores" in the class "Teachers" for the grades in all of the features.
	
</details>

<details>
<summary><b>The function "fake_teachers_algorithm"</b></summary> 
	
This function can be used to perform tests using fake labeled data. You can either use it with exsiting MIDI files of the fake data, or with the output of the function "generate_random_mistakes_data" that can be found in the file "Fake_Data.py".
You should psss the function the following parameters:
* ```from_midi_files_or_not``` - 'true' if you are using existing MIDI files, 'false' otherwise.
* ```number_of_teachers``` - number of different teachers that will grade the fake data.
* ```folder``` - optional. 'None' (default value) if you are not using existing MIDI files (in this case you must pass the parameter "performances_data"). If you are using existing MIDI files, pass the path of the folder contating them. 
* ```performances_data``` - optional. 'None' (default value) if you are using existing MIDI files (in this case you must pass the parameter "folder"). If you are not using existing MIDI files, pass the performances data (the output of the "generate_random_mistakes_data" function). 

**return value of the function** - a dictionary of <song_name>: <"Song" Class instance> for each song. 
	
	
</details>
