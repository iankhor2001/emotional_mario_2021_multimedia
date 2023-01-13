# Introduction
This is a project based on the MediaEval 2021 task _Emotional Mario: A Games Analytics Challenge_. In this project, we built a model to identify key game events in the game _Super Mario Bros_. Such key events include the end of a level, a power-up or extra life for Mario, or Mario’s death. 

The link to the detailed prompt is as follows: 
https://multimediaeval.github.io/editions/2021/tasks/emotionalmario/

# [Report](https://github.com/iankhor2001/emotional_mario_2021_multimedia/blob/master/Multimedia%20Final%20Report%20-%20signed.pdf)
Please feel free to read our [report](https://github.com/iankhor2001/emotional_mario_2021_multimedia/blob/master/Multimedia%20Final%20Report%20-%20signed.pdf) for more details of the project.

URL: https://github.com/iankhor2001/emotional_mario_2021_multimedia/blob/master/Multimedia%20Final%20Report%20-%20signed.pdf

# Instructions
Please put `number_dictionary.npy` into the same directory as `model.py`

Create the environment:
1. Environment setup
    1. For Windows OS: `conda env create -f mario_win.yml`
    1. For MAC OS: `conda env create -f mario_mac.yml`

1. Enter the environment:
`conda activate mario`

1. Run model:
`python3 model.py -i [input files directory path] -o [output file
path]`


## Note
The "ML_models" folder contains the machine learning models we have tried before. They are Jupyter Notebook so you can see the output for each cell. However, since they are not our mainly approaches, the related file path is not built which means you may not be able to run it. Those models are just for your information.

Since our mainly approach is not machine learning method, there is no pre-trained model. 
Please follow the instructions below to generate result file from model.py.
