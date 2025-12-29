import os
import numpy as np
import scipy.signal as ss
import tensorflow
import tensorflow as tf
from tensorflow import keras
import h5py
from matplotlib import colors
from time import time, sleep
import pickle
import pandas as pd
from datetime import datetime
from tensorflow.python.client import device_lib
from process_signal import load_experimental_data
curr_dir=os.path.dirname(os.path.abspath(__file__))
def run_inference(data_path,
        model_path,
        session,
        channel_sessions=None,
        export_spikes=True,
        continuous_prediction=False):

    
    channel = channel_sessions.get(session, None)
    if channel is None:
        raise ValueError(f"No channel mapping provided for session {session}")

    Fs = 1250 # Hz, sampling freq

    # Load signal + ripples
    filtered_signal, ripples = load_experimental_data(
        data_path,
        session,
        channel=channel,
        load_data=True,
        downsample=Fs,
        normalize=False,
        offset=0.16,
        scale_data=True
    ) # in microvolts

    filtered_signal/=1000.0 # to mV
    
    # Switch or reshaping into 1s segments, running 
    lfp = filtered_signal.reshape(-1)

    print('running Tensorflow v{}'.format(tf.__version__))
    print('running on devices:\n', device_lib.list_local_devices())
    print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
    print('GPU device:\n', tf.test.gpu_device_name())


    # load info on best model (path, threhsold settings)
    with open(os.path.join(curr_dir, os.pardir, 'best_model.pkl'), 'rb') as f:
        best_model = pickle.load(f)
        print(best_model)

    # some needed parameters
    # Fs = 1250 # Hz, sampling freq (moved to top)
    lag = int(100 * Fs / 1000) # 100 ms @ Fs

    # Threshold settings for detecting ripple events from prediction, 
    threshold = best_model['threshold'] # detection threshold on the interval (0, 1)
    distance = best_model['distance']  # timesteps, distance*Fs/1000 peak interdistance in units of ms
    width = best_model['width']       # timesteps, width*Fs/1000 peak width in units of ms. 

    # Container for predictions
    session_predictions = {}

    # load the 'best' performing model on the validation sets
    if model_path is not None:
        for model_file in os.listdir(model_path):
            if model_file.endswith('.h5'):
                model_file = os.path.join(model_path, model_file)
                print(f"Loading model from: {model_file}")
                model = keras.models.load_model(model_file)
    
                # remove the .h5 extension
                model_name=os.path.basename(model_file)[:-3]

                model.summary()
    
            # see scipy.signal.find_peaks documentation
            print(f"\n--- Running session: {session} ---")

            # Switch or reshaping input into segments, or running on full time series
            if continuous_prediction:
                # Predict using entire dataset at once
                Y_cont_pred = model.predict(np.expand_dims(np.expand_dims(lfp, 0), -1))
            else:
                # Reshape time axis to segments of Fs duration
                segment_length = int(0.5 * Fs) # Fs = 0.5s segments

                # run predictions n times with shifts of length segment_length / n,
                # then final output will be averaged
                n = 5 # nicely divisible with Fs=1250
                shift = int(segment_length / n)
                container = []
                for i in range(n):
                    lfp_reshaped = np.concatenate((np.zeros((1, i * shift, 1)), 
                                                np.expand_dims(np.expand_dims(lfp, 0), -1)), axis=1)
                    print ('shifted input shape:', lfp_reshaped.shape)
                    # pad with zeros 
                    lfp_reshaped = np.concatenate((lfp_reshaped, 
                                                    np.zeros((1, segment_length - 
                                                            (lfp_reshaped.size % segment_length), 1))), 
                                                    axis=1)
                    
                    # reshape into segments of length  
                    lfp_reshaped = lfp_reshaped.reshape((-1, segment_length, 1))
                    print('reshaped input shape:', lfp_reshaped.shape)
                    # run prediction on data
                    y_hat = model.predict(lfp_reshaped)
                    print('raw output shape:', y_hat.shape)
                    # Reshape to zero-padded size
                    y_hat = y_hat.reshape((1, -1, 1))[:, :lfp_reshaped.size, :]
                    print('reshaped output shape:', y_hat.shape)
                    # strip elements that were padded with zeros
                    container.append(y_hat[:, i * shift:i * shift + lfp.size, :])

                # average or median
                y_hat = np.median(container, axis=0).flatten()

                # remove intermediate predictions
                del container, lfp_reshaped

                predictions, _ = ss.find_peaks(y_hat, height=threshold, distance=distance, width=width) # in samples
                probabilities = y_hat[predictions]

                session_predictions[model_name] = {
                    'predictions': predictions,
                    'probabilities': probabilities
                }

    if export_spikes:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions")
        os.makedirs(out_dir, exist_ok=True)
        pkl_path = os.path.join(out_dir, f"{session}_predictions.pkl")
        
        with open(pkl_path, "wb") as f:
            pickle.dump(session_predictions, f)
        print(f"Saved predictions to: {pkl_path}")
