Since our mainly approach is not machine learning method, there is no pre-trained model. Please follow the instructions below to generate result file from model.py.

Instructions:
Please put number_dictionary.npy into the same directory as model.py

Create the environment:
[for windows system]: conda env create -f mario_win.yml
[for mac system]: conda env create -f mario_mac.yml

Enter the environment:
conda activate mario

Run model:
python3 model.py -i [input files directory path] -o [output file
path]


Note:
The "ML_models" folder contains the machine learning models we have tried before. They are Jupyter Notebook so you can see the output for each cell. However, since they are not our mainly approaches, the related file path is not built which means you may not be able to run it. Those models are just for your information.
