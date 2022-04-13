import logging
import pytest

timeout = 60
dimension = 128
delete_timeout = 60

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
log_name = f"knowhere_ci"
logging.basicConfig(filename=f"/tmp/{log_name}.log",
                    level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def pytest_addoption(parser):
    parser.addoption("--dataset", action="store", default="sift", help="public dataset for testing")
    parser.addoption("--entities", action="store", default=200000, help="num entities of data for testing")
    parser.addoption("--path", action="store", default="/home/data/milvus/raw_data/sift1b/", help="location of dataset")


@pytest.fixture
def dataset(request):
    return request.config.getoption("--dataset")


@pytest.fixture
def entities(request):
    return request.config.getoption("--entities")


@pytest.fixture
def path(request):
    return request.config.getoption("--path")

