#!/usr/bin/env python
# coding: utf-8

"""
- Model for Emotional Mario: A Games Analytics Challenge
- This script is for classifying important events in playthrough of 
  the game Super Mario Bro. based on the game frame image.
- Team 24 廖宏淇, 許育恩, 羅詔文
"""

############################# - Import Packages - #############################
# %matplotlib inline
import argparse
import pandas
import pickle
import numpy
import cv2
import csv
import os


############################# - File Operations - #############################

def count_files(path):
    # Count the amount of files (images) in a directory
    count = 0
    for root_dir, cur_dir, files in os.walk(path):
        count += len(files)
    return count


########################## - Get Dictionary Key functions - ########################

def get_key_dict(val):
    # Return key for value in dictionary
    val_bw = cv2.cvtColor(val, cv2.COLOR_BGR2GRAY)
    val_bin = numpy.interp(val_bw, (val_bw.min(), val_bw.max()), (0, +1))
    for key, dict_list in number_dictionary.items():
        for dict_val in dict_list:
            if numpy.array_equal(val_bin,dict_val):
                return True, key
    return False, 0


def get_current_time(current_frame, last_time, frame):
    # Return time of the frame
    d100 = frame[24:31,208:215]
    d10 = frame[24:31,216:223]
    d1 = frame[24:31,224:231]
    
    num1_valid, time_num1 = get_key_dict(d1)
    if not num1_valid:
        return last_time
    num10_valid, time_num10 = get_key_dict(d10)
    if not num10_valid:
        return last_time
    num100_valid, time_num100 = get_key_dict(d100)
    if not num100_valid:
        return last_time

    cur_time = int(f"{time_num100}{time_num10}{time_num1}")
    return cur_time


##################### - Check if the stage/score/time changed - #####################

def new_stage_check(current_frame, last_stage, frame):
    """
    - Check if the stage changed
    - Return False and last stage value when the number 
        is not identified
    - Return True when stage is change
    - The stage format is "x-x"
    """
    frame_stage_1 = frame[24:31,152:159]
    frame_stage_2 = frame[24:31,168:175]
    num1_valid, stage_num1 = get_key_dict(frame_stage_1)
    num2_valid, stage_num2 = get_key_dict(frame_stage_2)
    if not num1_valid&num2_valid:
        return False, last_stage
    if f"{stage_num1}-{stage_num2}" == last_stage:
        return False, last_stage
    else: 
        return True, f"{stage_num1}-{stage_num2}"

    
def new_score_check(current_frame, last_score, frame):
    """
    - Check if the score changed
    - Return False and last score value when the number 
        is not identified
    - Return True when score is change
    - The first and last two digits are not accounted into the 
        algorithm since they won't change
    - The stage format is "xxx"
    """
    d1 = frame[24:31,48:55]
    d10 = frame[24:31,40:47]
    d100 = frame[24:31,32:39]
    
    num1_valid, score_num1 = get_key_dict(d1)
    if not num1_valid:
        return False, last_score
    num10_valid, score_num10 = get_key_dict(d10)
    if not num10_valid:
        return False, last_score
    num100_valid, score_num100 = get_key_dict(d100)
    if not num100_valid:
        return False, last_score

    ## convert new_score and compare with previous score
    new_score = int(f"{score_num100}{score_num10}{score_num1}")
    if new_score == last_score:
        return False, last_score
    else: 
        return True, new_score
    
    
def status_down_check(frame_num, counting_time, function_counter, stage, frame):
    """
    - Check if the time is frozen
    - Return (flag_time, counting_time, counter, is_time_limit)
        flag_time is the event identifier,
        counting_time is the current game time
        counter is the counter for which the frame number in-game time lasted
        is_time_limit is the identifier for which the in-game time is frozen
    - flag_time:-
        return True if in-game time equals the stage time limit
        return True if the in-game time is frozen (counter exceed threshold)
        return False otherwise
    - is_time_limit: Return True if time is at limit, otherwise False
    """
    cur_time = get_current_time(frame_num, counting_time, frame)
    if (cur_time==stage_limit.get(stage)):
        function_counter=0
        return True, cur_time, function_counter, True
    if (cur_time==(stage_limit.get(stage)-1)):
        return False, cur_time, function_counter, False
    if (function_counter>TIME_THRESHOLD and cur_time==counting_time):
        function_counter=0
        return True, cur_time, function_counter, False
    elif (cur_time != counting_time):
        function_counter=0
        return False, cur_time, function_counter, False
    else:
        function_counter = function_counter+1
        return False, cur_time, function_counter, False


##################### - Load the dictionarys for operations - #####################

# Load dictionary for number from number_dictionary.npy
dictionary_file = open("./number_dictionary.npy", "rb")
number_dictionary = pickle.load(dictionary_file)
dictionary_file.close()

# Load dictionary for stage time limit
stage_limit = {
    '1-1': 400, '1-2': 400, '1-3': 300, '1-4': 300,
    '2-1': 400, '2-2': 400, '2-3': 300, '2-4': 300,
    '3-1': 400, '3-2': 300, '3-3': 300, '3-4': 300,
    '4-1': 400, '4-2': 400, '4-3': 300, '4-4': 400,
    '5-1': 300, '5-2': 400, '5-3': 300, '5-4': 300,
    '6-1': 400, '6-2': 400, '6-3': 300, '6-4': 300,
    '7-1': 400, '7-2': 400, '7-3': 300, '7-4': 400,
    '8-1': 400, '8-2': 400, '8-3': 400, '8-4': 300  ### to check
}

# Load event name dictionary
event_dictionary = {
    0: "new_stage",
    1: "flag_reached",
    2: "status_up",
    3: "status_down",
    4: "life_lost"
}

# Set Argument Parser
argument_parser = argparse.ArgumentParser(
    description="Emotional Matio Task for CS357000 Classification Model")

argument_parser.add_argument("-i", "--input-path", type=str, default=None)
argument_parser.add_argument("-o", "--output-path", type=str, default=None)


############################## - Main Operation - ##############################
# if __name__ == '__main__':
    
# Setup input/output path
args = argument_parser.parse_args()
dir_path = args.input_path
participant_name = dir_path.split("/")[-1]
if participant_name == "":
    participant_name = dir_path.split("/")[-2]

input_path = f"{dir_path}/{participant_name}_game_frame"
output_path = args.output_path

# Initiate variables and set initial values
current_frame = 1
current_score = 0
current_stage = "1-1"
TIME_THRESHOLD=35
is_time_limit_before = True
counter = 1
lock_counter = 0
counting_time=400
event_frame = []
event_list = []

# Start classification process
print(f"start classification")
for frame_number in range(1,count_files(input_path)):
    current_time = counting_time            

    # Read frame
    frame = cv2.imread(f'{input_path}/game_{frame_number}.png')

    # Checking Value Changes
    flag_score, new_score = new_score_check(frame_number, current_score, frame)
    flag_stage, next_stage = new_stage_check(frame_number, current_stage, frame)
    flag_time, counting_time, counter, is_time_limit = status_down_check(frame_number, counting_time, counter, current_stage, frame)

    # Predictions - Check Flags
    if flag_stage:
        current_stage = next_stage
        current_score = 0
        lock_counter = 50
        event_frame.append((frame_number-1, next_stage, 0, 1))
        event_frame.append((frame_number, next_stage, 0, 0))
        print(f"{frame_number-1},flag_reached")
        print(f"{frame_number},new_stage")

    elif flag_score:
        if new_score == 0 and is_time_limit:  ############# need to add time condition
            event_frame.append((frame_number-1, current_stage, new_score,4))
            print(f"{frame_number-1},life_lost")

        elif ((new_score-current_score) == 10):
            lock_counter = 0
            event_frame.append((frame_number-6, current_stage, new_score,2))
            print(f"{frame_number-6},status up")
        current_score = new_score

    elif flag_time:
        if (lock_counter > 100):
            if(counting_time == stage_limit.get(current_stage)):
                if not is_time_limit_before:
                        print(f"{frame_number},life_lost")
                        event_frame.append((frame_number-1, current_stage, new_score,4))
            else:
                lock_counter = 35
                event_frame.append((frame_number-35, current_stage, current_score,3))
                print(f"{frame_number-35},status down")

    lock_counter += 1
    is_time_limit_before = is_time_limit

# Outputing value into file
print(f"recording classification")

for frame in event_frame:
    event_list.append([frame[0],event_dictionary[frame[3]]])
    print(f'{frame[0]},{event_dictionary[frame[3]]}')

with open(f'{output_path}', 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(event_list)

print(f"end classification")
print(event_frame)