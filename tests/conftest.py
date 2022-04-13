import logging
import pytest


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
log_name = f"knowhere_ci"
logging.basicConfig(filename=f"/tmp/{log_name}.log",
                    level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def pytest_addoption(parser):
    parser.addoption("--data_type", action="store", default="sift", help="data_type for testing")
    parser.addoption("--entities", action="store", default=200000, help="num entities of data")
    parser.addoption("--path", action="store", default="/home/data/milvus/raw_data/sift1b/", help="test data location")


@pytest.fixture
def data_type(request):
    return request.config.getoption("--data_type")


@pytest.fixture
def entities(request):
    return int(request.config.getoption("--entities"))


@pytest.fixture
def path(request):
    return request.config.getoption("--path")

