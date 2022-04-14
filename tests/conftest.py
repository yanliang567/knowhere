import logging
import pytest


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
log_name = f"knowhere_ci"
logging.basicConfig(filename=f"/tmp/{log_name}.log",
                    level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def pytest_addoption(parser):
    parser.addoption("--data_type", action="store", default="sift", help="data_type for tests")
    parser.addoption("--entities", action="store", default=200000, help="num entities of data for latency tests")
    parser.addoption("--nq", action="store", default=10000, help="search vectors for latency tests")
    parser.addoption("--top_k", action="store", default=10, help="nearest vectors for latency tests")

    parser.addoption("--path", action="store", default="/home/data/milvus/raw_data/sift1b/", help="test data location")


@pytest.fixture
def data_type(request):
    return request.config.getoption("--data_type")


@pytest.fixture
def entities(request):
    return int(request.config.getoption("--entities"))


@pytest.fixture
def nq(request):
    return int(request.config.getoption("--nq"))


@pytest.fixture
def top_k(request):
    return int(request.config.getoption("--top_k"))


@pytest.fixture
def path(request):
    return request.config.getoption("--path")

