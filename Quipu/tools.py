# Project Quipu - tools for processing traces

import numpy as np
import pandas as pd
import glob



def noiseLevels(train = None):
    """
    Gives typical noise levels in the system 
    
    :param train: data to train on (numpy array)
    :return: typical noise levels (default: 0.006)
    """
    global constantTypicalNoiseLevels
    #constantTypicalNoiseLevels = 4
    if ~('constantTypicalNoiseLevels' in dir()):
        constantTypicalNoiseLevels = 0.006 # default
    if train is not None:
        tmp = np.array( list(map(np.std, train)) );
        constantTypicalNoiseLevels = tmp[~np.isnan(tmp)].mean()
    return constantTypicalNoiseLevels


def normaliseLength(trace, length = 600, trim = 0):
    """
    Normalizes the length of the trace and trims the front 
    
    :param length: length to fit the trace into (default: 600)
    :param trim: how many points to drop in front of the trace (default: 0)
    :return: trace of length 'length' 
    """
    if len(trace) >= length + trim:
        return trace[trim : length+trim]
    else:
        return np.append(
            trace[trim:],
            np.random.normal(0, noiseLevels(), length - len(trace[trim:]))
        )    


    
    
