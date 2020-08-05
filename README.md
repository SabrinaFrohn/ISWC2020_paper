# ISWC2020_paper
This repository contains the code and data used for analysis in the ISWC 2020 paper "Towards a Characterisation of Emotional Intend During Scripted Scenes Using In-Ear Movement Sensors". 
*note: While the study is reported in English the recordings have been done in German.*

## Abstract 

Theatre provides a unique environment in which to obtain detailed data on social interactions in a controlled and repeatable manner. This work introduces a method for capturing and characterising the underlying emotional intent of performers in a scripted scene using in-ear accelerometers.  
Each scene is acted with different underlying emotional intentions using the theatrical technique of Actioning. The goal of the work is to uncover characteristics in the joint movement patterns that reveal information on the positive or negative valence of these intentions. 
Preliminary findings over 3x18 (Covid-19 restricted) non-actor trials suggests people are more energetic and more in-sync when using positive versus negative intentions.

## Experiment Data

The folder "data" contains the sensor and audio data aquired during the experiment. The sensor data was cut from the original files in two ways. 

1. The trial data used for analysis in the paper: 
This data was cut from the original (synchronized) file by using button press information. The button was pressed in the Android app by the experimenter during the recording to indicate start and end of each trial. These timestamps were recorded in one app and used for both eSense sensor files. The analysis in the paper is based on these trials. They are numbered (01-20) and can be found in the folders for each eSense device (eSense_0237 or eSense_0308).

2. Improved trial data: 
During the cutting of the data for a successor study of the presented work it was found that the button presses do not provide 100% accurate timestamps. Instead the trial data was irregularily shifted a few seconds per trial. Therefore, the trial data was recut according to timestamps gathered from the audio recordings of the experiment by the eSense devices. Those files can be found in the "trials" folder for each eSense device (eSense_0237 or eSense_0308). Redoing the analysis showed only minimal differences in results but further analysis should be based on this data and studies should use this cutting method. 

The folder "timelines" contains excel files with the start and end timestamps of trials for each study. They were manually collected from the audio data and used to cut the audio and sensor data into trials. 

Lastly, each study folder contains an "experiment_00x_verbs.xls" file. Those files contain information about the transitive verbs used in each trial, experimenter notes and general information.  
