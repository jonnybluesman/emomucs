
import os
import glob

import librosa
import librosa.display
import pandas as pd
import numpy as np

# ensuring the reprod. of the exp setup
from numpy.random import seed
from numpy.random import randint
# seed(1992)

from joblib import Parallel, delayed  
from tqdm import tqdm


def get_offset_from_track(audio_file, clip_duration=30):
    """
    """
    
    dict_item = {
        'track_id': os.path.basename(audio_file).split(".")[0],
        'duration': librosa.get_duration(filename=audio_file)}

    dict_item['offset'] = \
        randint(0, np.ceil(dict_item['duration'] - clip_duration)) \
        if dict_item['duration'] > clip_duration else 0.0

    return dict_item


def get_offsets_from_dataset(audio_dir, clip_duration=30, njobs=1):
    """
    Returns a list with dict entries, each associated to a specific
    track. An inner dict contains the track id, the full duration
    of the corresponding track, and an offset (in seconds) chosen
    randomly to extract a clip of fixed "clip_duration".
    
    TODO:
        - Logging info and warnings...
    """
    
    audio_paths = glob.glob(audio_dir + "/*.mp3")
    
    audio_ofs = Parallel(n_jobs=njobs)\
        (delayed(get_offset_from_track)(audio_file, clip_duration) \
         for _, audio_file in zip(tqdm(range(len(audio_paths))), audio_paths))

    return audio_ofs


def save_track_metadata(track_mapping, save_path, track_id="track_id"):
    """
    Convert a list of dictionaries into a pandas dataframe, where the track id
    is the index of the dataframe, and save it as a csv file.
    """
    
    meta_df = pd.DataFrame(track_mapping)
    meta_df[track_id] = pd.to_numeric(meta_df[track_id])
    meta_df.set_index(track_id, inplace=True)
    meta_df.to_csv(save_path)
    
    return meta_df

def min_max_scaling(values, min_values=np.array([1., 1.]), max_values=np.array([9., 9.])):
    """
    Min-max scaling for ensuring that the annotations are in the [-1, 1] VA range.
    Minimum and maximum values for each column can be specified individually.
    """
    return 2 * ((values - min_values) / (max_values - min_values)) - 1


def get_duration_from_file(audio_file):
    """
    Simple wrapping function of ...
    
    """
    track_id = int(os.path.basename(audio_file).split('.')[0])
    return track_id, librosa.get_duration(filename=audio_file)


def compute_duration_per_track(audio_dir, njobs=1, save=None):
    
    audio_paths = glob.glob(audio_dir + "/*.mp3")
    print("Processing {} audio files.".format(len(audio_paths)))
    
    audio_dur = Parallel(n_jobs=njobs)\
        (delayed(get_duration_from_file)(audio_file) \
         for _, audio_file in zip(tqdm(range(len(audio_paths))), audio_paths))
    audio_dur = {track_id: duration for track_id, duration in audio_dur}
    
    if save is not None:
        with open(save, "wb") as f:
            joblib.dump(audio_dur, f)
            
    return audio_dur