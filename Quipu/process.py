# Project Quipu - processing traces

import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot

    
def getIndexedTraces(ds):
    "returns [(t,current)] for traces in ds"
    def addIndex(vec):
        "add index to a vector and zip"
        return np.array([np.arange(len(vec)), vec]).T
    return  np.vstack( ds.trace.apply(addIndex) )


def measureUnfoldedLevel(ds, verbose = False):
    """
    measure the current level of unfolded events
    """
    points = getIndexedTraces(ds)
    from sklearn.cluster import KMeans
    x = points[points[:,0] > 150, 1].reshape((-1,1))
    # remove outliers 
    std = np.std(x)
    mean = np.mean(x)
    x = x[x > mean - 4*std].reshape((-1,1)) 
    # ML clustering
    kmeans = KMeans(n_clusters=3, random_state=0).fit(x)
    x_cluster = kmeans.predict(x)
    means = [ np.mean(x[x_cluster == i]) for i in range(3)]
    means =  sorted(means) 
    level_one = means[1]
    if np.abs(level_one) > 0.35 or np.abs(level_one) < 0.1:
        print("Warning! Unfolded level detector in unexpected range: ",leven_one)
    if verbose: #feedback
        pyplot.figure()
        pyplot.hist2d(points[:,0], points[:,1], 
                 bins=(70*2, 50*2),
                 range = [[0, 700], [-0.45, 0.05]],
                 cmax = 100000/4 # clip max
                )
        pyplot.plot([0,700], [level_one]*2, 'r--')
    return level_one


def measureLenghtNormalisation(ds, aim_length):
    """
    measure how much average length has to adjusted to fit the aim length
    :param ds: dataset, should be split into independant experiments
    :param aim_length: the aimed length of traces. must manualy account for variation and zeros. 
                       example: for 800 total length pick aim_length=600
    :param unfolded_level: default level of current for this experiment
    :return: 
    """
    unfolded_level =  measureUnfoldedLevel(ds[ds.Filter])
    mean_area = ds[ds.Filter].area.mean()
    expected_length = np.abs( mean_area / unfolded_level )
    # could be an issue since we are bending and twisting the start position
    length_normalisation = aim_length / expected_length
    return length_normalisation
    
def normaliseTracesLength(traces, length_normalisation):
    """
    normalises the length 
    :param traces: traces as pandas series
    :param length_normalisation: multiplication factor 
    :return: the original with modified traces (not a copy)
    """
    import scipy.signal as signal
    return traces.apply(lambda x: signal.resample(x, int( len(x) * length_normalisation)))


def normaliseTracesMagnitude(traces, unfolded_level):
    """
    Normalise traces to be equal to one at unfolded level 
    Also try compressing the network 
    :param traces: traces as pandas series
    :param unfolded_level: default level of current for this experiment
    :return: normalised traces
    """ 
    return traces / np.abs(unfolded_level)


def normaliseTracesFromDataset(ds, aim_length, verbose=False):
    """
    Normalise entire dataset
    :param ds: dataset, should be split into independant experiments
    :param aim_length: the aimed length of traces. must manualy account for variation and zeros. 
                       example: for 800 total length pick aim_length=600
    :return: normalised traces as pandas series
    """
    experiments = ds.nanopore.unique()
    traces = ds.trace
    for nanopore in experiments:
        sel = ds.nanopore == nanopore
        data = ds[sel & ds.Filter]
        length_normalisation = measureLenghtNormalisation(data, aim_length)
        unfolded_level = measureUnfoldedLevel(data)
        ts0 = traces[sel]
        ts1 = normaliseTracesLength(ts0, length_normalisation)
        ts2 = normaliseTracesMagnitude(ts1, unfolded_level)
        traces[sel] = ts2
    return traces


    
#  outdated method
def normaliseTraces(ds, verbose=False):
    """
    Normalise traces to be equal to one at unfolded level 
    Also try compressing the network 
    "param ds: 
    :return: normalised traces
    """
    normalisation = measureUnfoldedLevel(ds)
    if verbose:
        points = getIndexedTraces(ds)
        pyplot.figure(figsize=(9, 5.6))
        pyplot.hist2d(points[:,0], points[:,1], 
                 bins=(70*2, 50*2),
                 range = [[0, 900], [-0.45, 0.05]],
                 cmax = 100000/2 # clip max
                );
        pyplot.plot([0,700], [normalisation]*2, "-r")
    
    return ds.trace.copy() / np.abs(normalisation)
    
#  outdated method
def normaliseDataset(ds, verbose=False):
    """
    Normalise entire dataset
    """
    experiments = ds.nanopore.unique()
    traces = []
    for nanopore in experiments:
        sel = ds.nanopore == nanopore
        traces.append( normaliseTraces(ds[sel]) )
    normalised_traces = pd.concat(traces)
    ds["trace"] = normalised_traces
    return ds
