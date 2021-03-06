# Piano-Preformance-Autograder
## Table of contents
* [General info](#general-info)
* [Installations](#installations)
* [GUI](#gui)
* [Run experiments with the data](#run-experiments-with-the-data)
* [Fake data & Fake teachers](#fake-data-and-fake-teachers)

## General info
This project is designed to make automatic evaluations of piano performances recorded with a MIDI keyboard.
The evaluation is given by using different Machine Learning algorithms.
Asside from getting a numerical grade for features of the performance, the players also get a recommendation for what should be their next step in their training.

## Installations
Follow the next instructions:
* Clone the project and open it using your local Python IDE.
* Install all required packages: 
   * Open Terminal/Command Line and go to the same directory where you saved the project's files (Alternatively - use the Terminal in your Python IDE, if this feature exists in your IDE).
   * Follow the instruction in this [link](https://cs.gmu.edu/~marks/112/projects/PlaySong.pdf) to install the package "pyaudio" (follow only the "pyaudio" installing explanation, you don't need to download anything else (although you might need to download "Homebrew" using this [link](https://brew.sh) if you are using MacOS)).
   * Run the command:
   ```
   pip3 install -r requirements.txt
   ```

## GUI
<details>
<summary><b>Using the GUI</b></summary>

After finishing the [installation process](#installations), in order to use the GUI follow these instructions:
* Run the file "GUI.py" (takes a few seconds on the first run).
* Enter a user's ID (must be 9 digits).
* Choose a Song Book (the first dropdown menu), and click on the "ok" button.
* Choose a song (the second dropdown menu), and click on the "confirm" button (a MIDI device must be connected at this point).
* The music sheet of the selected song will be presented, along with 3 buttons:
  * **"Record" button** - starts the recording session.
  * **"Try Again" button** - resets the recorded data and allows the user to start over the recording session. 
  * **"stop" button** - ends the recording session. After clicking on this button, the users will get their feedback (grades & next step recommendation). Then, the users will have three options:
    * Accept our recommendation for the next learning step (try and play the song we recommended with the tempo we recommended).
    * Try and play the same piece again to get another feedback.
    * End the current training session.

</details>

<details>
<summary><b>Adding new songs or Song Books</b></summary>
	
The project contains a directory with 43 different musical pieces, grouped as Song Books. 
To add new musical pieces:
* Go to ".../project directory/songs/"
* Choos a Song Book to add the song to. Create a new folder if you want to add a new Song Book.
* Add the following files for each song:
   *  <song_name>.ly
   *  <song_name>.midi
   *  <song_name>.png

Data recorded while using the app will be saved in the directory ".../students recordings".

</details>
	
	
## Run experiments with the data
<details>
<summary><b>Initializing a "Performance" class instance</b></summary>

	
All of the users' performances are being saved as instances of the class "Performance", that can be found in the file **"Performance_class.py"**.
To initialize an instance of this class you will have to pass the following parameters:
* ```path``` - the path of the MIDI file of the performance. If you would like to initialize an instance for a fake performance that do not have a MIDI file, pass an arbitrary string.
* ``` name ``` - the name of the song played.
* ```player_name``` - the name of the player. If you would like to initialize an instance for a fake performance pass an arbitrary string.
* ```original_path``` - the path of the MIDI file of the "perfect" performance.
* ```prettyMidiFile_performance``` - optional. This parameter is for using fake data. When using real data, pass None (it's also the default value). If the performance do not have a MIDI file (meaning it's fake), pass the prettyMIDI instance of the performance.
* ```prettyMidiFile_original``` - optional. This parameter is for using fake data. When using real data, pass None (it's also the default value). If the performance do not have a MIDI file (meaning it's fake), pass the prettyMIDI instance of the "perfect" performance.	

After initializing, a Performance class' instance will have few more fields:
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
	
After initializing, a Performance class' instance will have few more fields:
* ```perfroamcnes``` - a list that will contain all of this song's performances. 
* ```fake_performances``` - a list that will contain all of this song's fake performances. 
	
</details>


## Fake data and Fake teachers

### Fake data
You can generate fake users' performances data based on existing songs. This algorithm inserts random noise to each MIDI feature.

<details>
<summary><b>Generating fake performances' data</b></summary>
	
Use the function **"generate_random_mistakes_data"** in the **"Fake_Data".py"** file, with the following parameters:
* ```folder``` - the path of the folder that contains the songs you would like to generate fake data based on. 
* ``` n ``` - number of fake performances generated out of **every song in the directory**. (For example - if n=3 and the directory contains 4 songs - 12 fake performances will be generated).
* ```create_midi_files``` - 'true' if you want to create a MIDI file for each fake performance you generate. 'false' otherwise (in this case you will have to use the return value of this function if you would like to use the fake generated data outside of this function).
* ```max_noise``` - a number between 0 to 1. Higher number will raise the magnitude ceiling for each change of the original song.
* ```max_percentage``` - a number between 0 to 1.  Higher number will raise the ceiling of the percentage of the song notes to be cahnged.
* ```min_noise``` - a number between 0 to 1.  Higher number will raise the magnitude floor for each change of the original song.
* ```min_percentage``` - a number between 0 to 1.  Higher number will raise the floor of the percentage of the song notes to be cahnged. 

**return value of the function** - the functions returns a list of instances of the class "Song_Class". Each instance represents a song, and all its fake performances are in the field 'fake_performances'.

</details>

### Fake teachers
We created a fake teachers algorithm that can grade performances and give a next step reccomendation. 

<details>
<summary><b>Create fake teachers</b></summary>

	
Use the function **create_fake_teachers** in the **"Automated_teacher.py"** file, with the following parameters:
* ```number_of_teachers``` - number of different fake teachers to be created.

**return value of the function** - the function returns a list of "Teacher" class instances, representing the different teachers created.
	
</details>

<details>
<summary><b>Grade performances using the fake teachers</b></summary>

	
Use the function **fake_teachers_feedback** in the **"Automated_teacher.py"** file, with the following parameters:
* ```performance``` - the performance that you want to be graded by the fake teachers, given as a "Performance" class instance.
* ```teachers``` - a list containing "Teacher" class' instances for each fake teacher (the output of the function "create_fake_teahcers").
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
You can adjust the fake teachers' grading & next step algorithms, by adjusting one or more of the following:
* **Unique values of each teacher** - you can set the range in which the unique value is being randomly selected from, for each feature seperatly. The ranges can be found in the function "create_fake_teachers" in the "Automated_teacher.py" file:
  * The variable "featureName_unique_next_step" - for the next step algorithm.
  * The variable "featureName_unique_score" - for the grades algorithm.
* **Different thresholds** - you can change the thresholds in one or more algorithms. The algorithms can be found in the following functions:
  * The function "give_next_step_recco" of the class "Teacher" for the next step reccomendations.
  * The function "give_scores" of the class "Teacher" for the grades in all of the features.
	
</details>

<details>
<summary><b>The function "fake_teachers_algorithm"</b></summary> 
	
This function can be used to perform tests using fake labeled data. You can either use it with exsiting MIDI files of the fake data, or with the output of the function "generate_random_mistakes_data" that can be found in the file "Fake_Data.py".
You should psss the function the following parameters:
* ```from_midi_files_or_not``` - 'true' if you are using existing MIDI files, 'false' otherwise.
* ```number_of_teachers``` - number of different fake teachers that you would like to grade the fake data (the fake teachers will be created as part of the run of the function).
* ```folder``` - optional. 'None' (default value) if you are **not** using existing MIDI files (in this case you must pass the parameter "performances_data"). If you are using existing MIDI files, pass the path of the folder contating them. 
* ```performances_data``` - optional. 'None' (default value) if you are using existing MIDI files (in this case you must pass the parameter "folder"). If you are **not** using existing MIDI files, pass the performances' data (the output of the "generate_random_mistakes_data" function). 

	**return value of the function** - a dictionary of <song_name>: <"Song" Class instance> for each song, such that each song contains all of the relevant labeled fake performances. 
	
	
</details>
