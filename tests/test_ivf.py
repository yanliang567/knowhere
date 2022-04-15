import logging
import pytest
import knowhere
import numpy as np
import json
import time
import utils


class TestIvf:
    @pytest.mark.latency
    @pytest.mark.parametrize("index", ["ivf", "ivfsq"])
    @pytest.mark.parametrize("nlist", [1024])
    @pytest.mark.parametrize("nprobe", [16])
    @pytest.mark.parametrize("metric_type", ["L2"])
    def test_ivfsq_latency(self, data_type, entities, index,
                           nq, top_k, nlist, nprobe, metric_type, path):
        logging.info(f"********** test latency start: data_type: {data_type}, entities: {entities} **********")
        dim = utils.dim(data_type)
        if index == "ivfsq":
            idx = knowhere.IVFSQ()
        elif index == "ivf":
            idx = knowhere.IVF()
        else:
            raise Exception("index: %s not supported in this test" % index)
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
                    "k": top_k,
                    "nlist": nlist,
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
        query_data = knowhere.ArrayToDataSet(
            arr[:nq, :]
            # np.random.uniform(1, 5, (1000, 128)).astype("float32")
        )
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
        assert(tt < 10)
        logging.info(idx)
        logging.info(dis)

    @pytest.mark.recall
    @pytest.mark.parametrize("index", ["ivf", "ivfsq"])
    @pytest.mark.parametrize("nlist", [1024])
    @pytest.mark.parametrize("nprobe", [16])
    def test_ivfsq_recall(self, data_type, index, nlist, nprobe, path):
        # recall uses fixed entities
        logging.info(f"********** test recall start: data_type:{data_type}， nlist:{nlist}, nprobe:{nprobe} **********")

        # dataset and search params are fixed for recall tests
        dataset = utils.get_dataset(data_type)
        logging.info(f"hdf5 dataset: {dataset}")
        metric_type = "L2"
        nq = 10000
        top_k = 10
        # insert_vectors = utils.normalize(metric_type, np.array(dataset["train"]))
        # query_vectors = utils.normalize(metric_type, np.array(dataset["test"][:nq]))
        insert_vectors = np.array(dataset["train"])
        query_vectors = np.array(dataset["test"][:nq])

        true_ids = np.array(dataset["neighbors"])

        # index
        dim = utils.dim(data_type)
        if index == "ivfsq":
            idx = knowhere.IVFSQ()
        elif index == "ivf":
            idx = knowhere.IVF()
        else:
            raise Exception("index: %s not supported in this test" % index)
        logging.info(f"data entities: {len(insert_vectors)}")
        data = knowhere.ArrayToDataSet(insert_vectors)
        cfg = knowhere.CreateConfig(
            json.dumps(
                {
                    "dim": dim,
                    "k": top_k,
                    "nlist": nlist,
                    "nprobe": nprobe,
                    "metric_type": metric_type,
                    "SLICE_SIZE": 4,
                }
            )
        )
        # start building index
        t0 = time.time()
        idx.Train(data, cfg)
        idx.AddWithoutIds(data, cfg)
        tt = time.time() - t0
        logging.info(f"index time: {tt}")
        # start searching
        for npb in [16, 32, 64]:
            cfg = knowhere.CreateConfig(
                json.dumps(
                    {
                        "dim": dim,
                        "k": top_k,
                        "nlist": nlist,
                        "nprobe": npb,
                        "metric_type": metric_type,
                        "SLICE_SIZE": 4,
                    }
                )
            )
            query_data = knowhere.ArrayToDataSet(query_vectors)
            t0 = time.time()
            ans = idx.Query(query_data, cfg, knowhere.EmptyBitSetView())
            tt = time.time() - t0
            logging.info(f"search time: {tt}")
            # retrieve search results
            result_ids = np.zeros((nq, top_k), np.int32)
            result_dis = np.zeros((nq, top_k), np.float32)
            t0 = time.time()
            knowhere.DumpResultDataSet(ans, result_dis, result_ids)
            tt = time.time() - t0
            logging.info(f"covert to dataset time: {tt}")
            assert (tt < 10)
            logging.info(f"result ids: {result_ids}")
            logging.info(f"true ids: {true_ids}")
            acc_value = utils.get_recall_value(true_ids[:nq, :top_k].tolist(), result_ids)
            logging.info(f"acc: {acc_value}")

            true_dis = np.array(dataset["distances"])
            logging.info(f"result dis: {result_dis}")
            logging.info(f"true dis: {true_dis}")

