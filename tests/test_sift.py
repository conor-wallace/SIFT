import os
import sys
import pytest
import logging
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.getcwd()))
from sift.learners import SIFTLightning


@pytest.fixture
def config_path():
    return "../sift/conf/sift_config.yaml"


@pytest.fixture
def config(config_path):
    return OmegaConf.load(config_path)


def test_sift(config):
    logging.info(config)
    _ = SIFTLightning(config)
