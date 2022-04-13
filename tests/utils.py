import os
import logging
import sklearn
import numpy as np
import h5py

VECTORS_PER_FILE = 1000000
SIFT_VECTORS_PER_FILE = 100000
BINARY_VECTORS_PER_FILE = 2000000

RAW_DATA_DIR = "/home/data/milvus/raw_data/"
HDF5_DATA_DIR = "/home/data/milvus/ann_hdf5/"

SIFT_SRC_DATA_DIR = RAW_DATA_DIR + "sift1b/"
DEEP_SRC_DATA_DIR = RAW_DATA_DIR + 'deep1b/'
JACCARD_SRC_DATA_DIR = RAW_DATA_DIR + 'jaccard/'
HAMMING_SRC_DATA_DIR = RAW_DATA_DIR + 'hamming/'

SIFT_HDF5 = HDF5_DATA_DIR + "sift-128-euclidean.hdf5"

FILE_PREFIX = "binary_"


def dim(data_type):
    if data_type == "sift":
        return 128
    elif data_type == "deep":
        return 96
    elif data_type == "jaccard":
        return 512
    else:
        return 0


def gen_file_name(idx, data_type):
    s = "%05d" % idx
    fname = FILE_PREFIX + str(dim(data_type)) + "d_" + s + ".npy"
    if data_type == "sift":
        fname = SIFT_SRC_DATA_DIR + fname
    elif data_type == "deep":
        fname = DEEP_SRC_DATA_DIR + fname
    elif data_type == "jaccard":
        fname = JACCARD_SRC_DATA_DIR + fname
    elif data_type == "hamming":
        fname = HAMMING_SRC_DATA_DIR + fname
    # elif data_type == "sub" or data_type == "super":
    #     fname = STRUCTURE_SRC_DATA_DIR + fname
    return fname


def get_len_vectors_per_file(data_type):

    if data_type == "sift":
        vectors_per_file = SIFT_VECTORS_PER_FILE
    # elif data_type in ["binary"]:
    #     vectors_per_file = BINARY_VECTORS_PER_FILE
    # elif data_type == "local":
    #     vectors_per_file = SIFT_VECTORS_PER_FILE
    else:
        raise Exception("data_type: %s not supported" % data_type)
    return vectors_per_file


def normalize(metric_type, X):
    if metric_type == "ip":
        logging.info("Set normalize for metric_type: %s" % metric_type)
        X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')
        X = X.astype(np.float32)
    elif metric_type == "l2":
        X = X.astype(np.float32)
    elif metric_type in ["jaccard", "hamming", "sub", "super"]:
        tmp = []
        for item in X:
            new_vector = bytes(np.packbits(item, axis=-1).tolist())
            tmp.append(new_vector)
        X = tmp
    return X


def get_dataset(data_type):
    """ Determine whether hdf5 file exists, and return the content of hdf5 file """
    if data_type == "sift":
        dataset = h5py.File(SIFT_HDF5)
    else:
        raise Exception("data_type: %s not supported" % data_type)
    return dataset

