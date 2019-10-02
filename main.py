######################################################################################### 

# Video Preprocessing Framework
# Matt Rochford

# This is the main function that implements the lip detector on each video 
# file and stores the output numpy arrays. Also includes all helper functions.

######################################################################################### 

# Import modules
from lip_detector import lip_detector
import os
import numpy as np

######################################################################################### 

# Define Paths
predictor_path = '/home/matt/thesis/preprocessing/shape_predictor_68_face_landmarks.dat' 
input_folder = '../LRW/lipread_mp4/' 
output_folder = '../LRW/'

#########################################################################################

# This function applies the lip detector function to all the files in the input folder
# and saves the output numpy files to the output folders.

def process_videos():

    # Loop over each file in entire dataset

    # Loops over each word directory Ex. ABOUT, ABSOLUTELY, ACCESS, etc.
    for directory in os.listdir(input_folder):

        # Loops over each sub directory, test train val
        for sub_directory in os.listdir(input_folder+directory):
            
            # Loops over each file in final directory
            for file in os.listdir(input_folder+directory+'/'+sub_directory):
                
                # Only use mp4 files
                if file.endswith('.mp4'):

                    print('Processing '+file)

                    # Generate label for output data file
                    label = os.path.splitext(file)[0] # Grab file base name
                    label = label + '.npy'

                    # Form full input and output paths using directory and individual file
                    path = input_folder+directory+'/'+sub_directory+'/'

                    # Perform lip detection and return data array
                    data_array = lip_detector(path+file)
                    
                    # If face was detected
                    if data_array is not None:

                        # Define path to output folder
                        path = output_folder+sub_directory+'/'

                        # Save output numpy array to output folder
                        np.save(path+label,data_array)


#########################################################################################

# This label create labels for each class and saves them as .npy files to the proper
# directory.

def label_maker():

    # Initialize counter for label making
    i = 0

    # Loops over each word directory Ex. ABOUT, ABSOLUTELY, ACCESS, etc.
    for directory in os.listdir(input_folder):

        # Hard code label of size 500 for each word in dataset
        label = np.zeros((500), dtype=int)

        # Assign value of 1 at proper location
        label[i] = 1

        # Create name for label file
        name = directory+'_label.npy'     

        # Create path to labels folder   
        path = output_folder+'labels/'

        # Save numpy file to labels folder
        np.save(path+name,label)

        # Increment counter
        i = i + 1

#########################################################################################

# This function is used to calculate max min and average word duration for help with 
# training. Average word duration is used to determine amount of frames to include for
# training.

def duration():

    # Initialize array to store word durations
    duration = []

    # Loop over each file in entire dataset

    # Loops over each word directory Ex. ABOUT, ABSOLUTELY, ACCESS, etc.
    for directory in os.listdir(input_folder):

        # Loops over each sub directory, test train val
        for sub_directory in os.listdir(input_folder+directory):
            
            # Loops over each file in final directory
            for file in os.listdir(input_folder+directory+'/'+sub_directory):
                
                # Only read txt files
                if file.endswith('.txt'):

                    data = [] # Initialize dummy variable to store text data
                    path = input_folder+directory+'/'+sub_directory+'/'+file # Define path to file

                    with open(path,'rt') as myfile:

                        for line in myfile:
                            data.append(line) # Append lines to dummy variable

                    time = float(data[-1].split()[1]) # Grab duration value
                    duration.append(time) # Add duration to total array

    print('Mean, max, and min word durations:')
    print(np.mean(duration))
    print(np.max(duration))
    print(np.min(duration))
 
#########################################################################################                   


if __name__ == "__main__":
    #duration()
    label_maker()
    process_videos()
    
