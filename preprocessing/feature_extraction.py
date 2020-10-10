
import os
import glob
import joblib

import pandas as pd
import numpy as np
import librosa

from joblib import Parallel, delayed  
from tqdm import tqdm


def post_process_separated_track(stereo_track, target_sr=22050, target_duration=30, padding=True):
    """
    Post-processing audio sequences obtained from the separation of tracks with demucs.
    """
    mono_track = librosa.to_mono(stereo_track) # from stereo to mono
    monores_track = librosa.resample(mono_track, 44100, target_sr) # resampling
    
    if padding and len(monores_track) < target_sr*target_duration:
        monores_track = librosa.util.fix_length(
            monores_track, target_sr*target_duration)

    return monores_track


def extract_audio_ts(audio_file, offset_df, clip_duration=30, sr=22050, padding=True):
    """
    
    We assume that the base name of the audio file path corresponds to the
    integer id that is associated to an entry in the offset_df dataframe.
    
    TODO:
        - The offset parameter should be either a dict or a dataframe;
        - Provide different types of padding (right, center, left);
    """
    
    track_id = int(os.path.basename(audio_file).split('.')[0])
    
    y, _ = librosa.load(
        audio_file, offset=offset_df.loc[track_id]["offset"], 
        duration=clip_duration, sr=sr)
    
    if padding and offset_df.loc[track_id]["duration"] < clip_duration:
        y = librosa.util.fix_length(y, sr*clip_duration)
        
    return track_id, y


def extract_audio_ts_from_dataset(audio_dir, offset_df, clip_duration=30, sr=22050, njobs=1, save=None):
    """
    Extracting audio time series from a collection of audio files.
    
    TODO:
        - same considerations of the atomic function used here;
        - implement a blacklist filtering mechanism;
    """
    
    audio_paths = glob.glob(audio_dir + "/*.mp3")
    
    audio_ts = Parallel(n_jobs=njobs)(delayed(extract_audio_ts)(audio, offset_df, clip_duration, sr) 
                                   for _, audio in zip(tqdm(range(len(audio_paths))), audio_paths))
    audio_ts = {track_id: track_audio_ts for track_id, track_audio_ts in audio_ts}
    
    if save is not None:
        with open(save, "wb") as f:
            joblib.dump(audio_ts, f, compress=3)
            
    return audio_ts


def extract_melspectogram(track_id, audio_ts, sr=22050, num_mels=40, fft_size=1024, win_size=1024, hop_size=512):
    """
    ... TODO
    
    Requiring the track_id is just a temporary workaround to a joblib's bug (does not 
    always keep the order of elements processed in parallel).
    """
    
    ms = librosa.feature.melspectrogram(
        y=audio_ts, sr=sr, n_mels=num_mels, 
        n_fft=fft_size, win_length=win_size, hop_length=hop_size)
    ms_db = librosa.power_to_db(ms, ref=np.max)
    
    return track_id, ms, ms_db


def extract_multisource_melspectogram(track_id, multi_audio_ts, sr=22050, num_mels=40, fft_size=1024, win_size=1024, hop_size=512):
    """
    ... TODO
    
    Same of the method before but for multi-sourced audio.
    """
    multi_source_mel = {}
    multi_source_lmel = {}
    
    for source_name, source_audio_ts in multi_audio_ts.items():
        _, ms, ms_db = extract_melspectogram(
            None, source_audio_ts, sr=sr, num_mels=num_mels, 
            fft_size=fft_size, win_size=win_size, hop_size=hop_size)
        
        multi_source_mel[source_name] = ms
        multi_source_lmel[source_name] = ms_db
    
    return track_id, multi_source_mel, multi_source_lmel


def extract_melspectograms_from_dataset(
    audio_ts_map, sr=22050, num_mels=40, fft_size=1024, win_size=1024, hop_size=512, njobs=1, save=None, compression=3):
    """
    
    Given a mapping {track_id : audio_ts} as a dict, compute the mel and the log-mel spectograms 
    for each track and return the resulting features as two separate items (indexed by track_id).
    """
    extract_fn = extract_melspectogram
    if isinstance(audio_ts_map[list(audio_ts_map.keys())[0]], dict):
        extract_fn = extract_multisource_melspectogram
    
    all_mels = Parallel(n_jobs=njobs)(delayed(extract_fn)(
        track_id, audio_ts, sr, num_mels, fft_size, win_size, hop_size) \
            for _, (track_id, audio_ts) in zip(tqdm(range(len(audio_ts_map))), audio_ts_map.items()))
    
    result_mel = {track_id: mel for track_id, mel , _ in all_mels}
    result_lmel = {track_id: logmel for track_id, _ , logmel in all_mels}
    
    if save is not None:
        with open(save, "wb") as f:
            joblib.dump({'mel': result_mel, 'lmel': result_lmel}, f)
            
    return result_mel, result_lmel