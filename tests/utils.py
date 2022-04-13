
VECTORS_PER_FILE = 1000000
SIFT_VECTORS_PER_FILE = 100000
BINARY_VECTORS_PER_FILE = 2000000

RAW_DATA_DIR = "/home/data/milvus/raw_data/"
SIFT_SRC_DATA_DIR = RAW_DATA_DIR + "sift1b/"
DEEP_SRC_DATA_DIR = RAW_DATA_DIR + 'deep1b/'
JACCARD_SRC_DATA_DIR = RAW_DATA_DIR + 'jaccard/'
HAMMING_SRC_DATA_DIR = RAW_DATA_DIR + 'hamming/'

FILE_PREFIX = "binary_"


def dim(dataset):
    if dataset == "sift":
        return 128
    elif dataset == "deep":
        return 96
    elif dataset == "jaccard":
        return 512
    else:
        return 0


def gen_file_name(idx, dataset):
    s = "%05d" % idx
    fname = FILE_PREFIX + str(dim(dataset)) + "d_" + s + ".npy"
    if dataset == "sift":
        fname = SIFT_SRC_DATA_DIR + fname
    elif dataset == "deep":
        fname = DEEP_SRC_DATA_DIR + fname
    elif dataset == "jaccard":
        fname = JACCARD_SRC_DATA_DIR + fname
    elif dataset == "hamming":
        fname = HAMMING_SRC_DATA_DIR + fname
    # elif data_type == "sub" or data_type == "super":
    #     fname = STRUCTURE_SRC_DATA_DIR + fname
    return fname


def get_len_vectors_per_file(dataset):

    if dataset == "sift":
        vectors_per_file = SIFT_VECTORS_PER_FILE
    # elif dataset in ["binary"]:
    #     vectors_per_file = BINARY_VECTORS_PER_FILE
    # elif dataset == "local":
    #     vectors_per_file = SIFT_VECTORS_PER_FILE
    else:
        raise Exception("data_type: %s not supported" % dataset)
    return vectors_per_file
