import logging
import pytest
import knowhere
import numpy as np
import json
import time
import utils


class TestIvfSq:
    @pytest.mark.latency
    def test_ivfsq_latency(self, data_type, entities, path):
        dim = utils.dim(data_type)
        idx = knowhere.IVFSQ()
        # arr5 = np.random.uniform(1, 5, (20000, 128)).astype("float32")
        # arr = np.append(arr, arr5, axis=0)
        vectors_per_file = utils.get_len_vectors_per_file(data_type)
        i = 0
        file = utils.gen_file_name(i, data_type)
        arr = np.load(file)
        i = 1
        while i < (entities // vectors_per_file):
            file = utils.gen_file_name(i, data_type)
            # file = "/home/data/milvus/raw_data/sift1b/binary_128d_00000.npy"
            arr = np.append(arr, np.load(file), axis=0)
            i += 1
        arr = np.float32(arr)
        logging.info(f"data entities: {len(arr)}")
        data = knowhere.ArrayToDataSet(arr)

        cfg = knowhere.CreateConfig(
            json.dumps(
                {
                    "dim": dim,
                    "k": 10,
                    "nlist": 100,
                    "nprobe": 4,
                    "metric_type": "L2",
                    "SLICE_SIZE": 4,
                }
            )
        )
        t0 = time.time()
        idx.Train(data, cfg)
        idx.AddWithoutIds(data, cfg)
        tt = time.time() - t0
        logging.info(f"index time: {tt}")
        query_data = knowhere.ArrayToDataSet(
            arr[:1000, :]
            # np.random.uniform(1, 5, (1000, 128)).astype("float32")
        )
        t0 = time.time()
        ans = idx.Query(query_data, cfg, knowhere.EmptyBitSetView())
        tt = time.time() - t0
        logging.info(f"search time: {tt}")
        idx = np.zeros((1000, 10), np.int32)
        dis = np.zeros((1000, 10), np.float32)
        t0 = time.time()
        knowhere.DumpResultDataSet(ans, dis, idx)
        tt = time.time() - t0
        logging.info(f"covert to data_type time: {tt}")
        assert(tt < 10)
        logging.info(idx)
        logging.info(dis)

    @pytest.mark.recall
    def test_ivfsq_recall(self, data_type, path):
        # recall uses fixed entities
        dataset = utils.get_dataset(data_type)
        logging.info(f"hdf5 dataset: {dataset}")
        metric_type = "L2"
        nq = 10000
        top_k = 10
        insert_vectors = utils.normalize(metric_type, np.array(dataset["train"]))
        query_vectors = utils.normalize(metric_type, np.array(dataset["test"][:nq]))
        true_ids = np.array(dataset["neighbors"])

        dim = utils.dim(data_type)
        idx = knowhere.IVFSQ()
        logging.info(f"data entities: {len(insert_vectors)}")
        data = knowhere.ArrayToDataSet(insert_vectors)

        nprobe = 4
        cfg = knowhere.CreateConfig(
            json.dumps(
                {
                    "dim": dim,
                    "k": top_k,
                    "nlist": 1024,
                    "nprobe": nprobe,
                    "metric_type": metric_type,
                    "SLICE_SIZE": 4,
                }
            )
        )
        t0 = time.time()
        idx.Train(data, cfg)
        idx.AddWithoutIds(data, cfg)
        tt = time.time() - t0
        logging.info(f"index time: {tt}")
        query_data = knowhere.ArrayToDataSet(query_vectors)
        t0 = time.time()
        ans = idx.Query(query_data, cfg, knowhere.EmptyBitSetView())
        tt = time.time() - t0
        logging.info(f"search time: {tt}")
        idx = np.zeros((nq, top_k), np.int32)
        dis = np.zeros((nq, top_k), np.float32)
        t0 = time.time()
        knowhere.DumpResultDataSet(ans, dis, idx)
        tt = time.time() - t0
        logging.info(f"covert to data_type time: {tt}")
        assert (tt < 10)
        logging.info(f"result ids: {idx}")
        logging.info(f"true ids: {true_ids}")
        # logging.info(dis)
