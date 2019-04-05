import os
import os.path
import zipfile
import cv2
import numpy as np
import random 
import shutil 

DIR_NOTEBOOKS = os.getcwd()
DIR_MAGIC_MIRROR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DIR_DATA = os.path.join(DIR_MAGIC_MIRROR, 'data')
DIR_DATA_RAW = os.path.join(DIR_DATA, 'raw')
DIR_DATA_INTERIM = os.path.join(DIR_DATA, 'interim')
DIR_DATA_PROCESSED = os.path.join(DIR_DATA, 'processed')

print("Current dir: ", DIR_NOTEBOOKS)
print("Parent dir: ", DIR_MAGIC_MIRROR)
print("Data dir:", DIR_DATA)
print("Raw Data dir:", DIR_DATA_RAW)
print("Interim Data dir:" , DIR_DATA_INTERIM)
print("Processed Data dir: ", DIR_DATA_PROCESSED)

def clean_folder(path): 
    "Deletes and recreates folder located at path."
    print("Deleting {}".format(DIR_DATA_INTERIM))
    shutil.rmtree(path)
    print("Recreating {}".format(DIR_DATA_INTERIM))
    os.mkdir(path)

raw_filename = 'video.zip'
print("Extracting: {}".format(os.path.join(DIR_DATA_RAW, raw_filename)))
raw_path = os.path.join(DIR_DATA_RAW, raw_filename)

with zipfile.ZipFile(raw_path, 'r') as raw_zip:
    raw_zip.extractall(DIR_DATA_INTERIM)

def get_video_length(path):
    """Given path to video, returns the amount of frames
    
    Parameters 
    ---------- 
    path : str
        Path to video file.
    
    Returns 
    ------- 
    int
        Number of frames in video.
    
    """
    # Create video capture object
    cap = cv2.VideoCapture(video_path) 

    # Get frame count properties 
    FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Release object
    cap.release()    
    
    return FRAME_COUNT

DIR_DATA_INTERIM_VIDS = os.path.join(DIR_DATA_INTERIM, 'video')
interim_files_all = [file for file in os.listdir(DIR_DATA_INTERIM_VIDS) if file.endswith('avi')]

print("Removing videos that are < 10 frames.")
print("Length before: ", len(interim_files_all))
for filename in interim_files_all:
    video_path = os.path.join(DIR_DATA_INTERIM_VIDS, filename)
    video_length = get_video_length(video_path)
    if video_length < 10: 
        interim_files_all.remove(filename)
        
interim_files = interim_files_all

print("Length after: ", len(interim_files_all))

def crop_video(video_path, crop_size=224): 
    """Given path to video, rescales and takes a 224x224 center crop of the video. 
    
    Parameters 
    ---------- 
    video_path : str
        Path to video file.
    crop_size : int 
        Size of center crop to take. Defaults to 224. 
    
    Returns 
    ------- 
    np.array 
        returns cropped np.array of the video
    """
    # Open video capture 
    cap = cv2.VideoCapture(video_path) 

    # Get video properties 
    FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialise return array 
    ret_arr = np.empty((0, crop_size, crop_size, 3))

    for i in range(FRAME_COUNT):
        # Read in video frame by frame 
        success, frame = cap.read() 
        
        # Resize image 
        if FRAME_WIDTH > FRAME_HEIGHT:
            scale = float(crop_size) / float(FRAME_HEIGHT)
            dim = (int(FRAME_WIDTH*scale+1), crop_size) 
            frame = np.array(cv2.resize(np.array(frame), dim)).astype(np.float32) # dim = (w,h)
        else:
            scale = float(crop_size) / float(FRAME_WIDTH)
            dim = (crop_size, int(FRAME_HEIGHT*scale+1))
            frame = np.array(cv2.resize(np.array(frame), dim)).astype(np.float32) 
            
        # Take center crop (224x224 by default)
        crop_x = int((frame.shape[0] - crop_size) / 2)
        crop_y = int((frame.shape[1] - crop_size) / 2)
        frame = frame[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        frame = np.reshape(frame, (1, crop_size, crop_size, 3))
        ret_arr = np.concatenate((ret_arr, frame), axis=0)

    ret_arr = np.expand_dims(ret_arr, axis=0)
    
    # Release object
    cap.release()
    
    return ret_arr

def convert_video_to_numpy(video_path, out_dir_path): 
    """Given path to video './data/thing/vid.avi', converts video file into .npy object and stores in out_dir_path. 
    
    Parameters 
    ---------- 
    video_path : str
        Path to video file.
    out_dir_path : str
        Path to output dir. 
    
    Returns 
    ------- 
    int
        Returns 0 on success and 1 on failure. 
    """
    npy_filename = video_path.split('/')[-1].split('.')[0] + '.npy'
    out_path = os.path.join(out_dir_path, npy_filename)
    arr = crop_video(video_path, crop_size=224)
    
    try: 
        np.save(out_path, arr)
        print("Saving numpy object at {}".format(out_path))
    except:
        return 1 
    
    return 0

random.shuffle(interim_files)
#interim_files_short = interim_files[:100]

# Crop files and output to out_path 
for filename in interim_files_short: 
    video_path = os.path.join(DIR_DATA_INTERIM_VIDS, filename)
    out_dir_path = DIR_DATA_PROCESSED
    convert_video_to_numpy(video_path, out_dir_path)

print("Deleting {}".format(DIR_DATA_INTERIM))
shutil.rmtree(DIR_DATA_INTERIM)
print("Recreating {}".format(DIR_DATA_INTERIM))
os.mkdir(DIR_DATA_INTERIM)


