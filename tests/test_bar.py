import os
import sys
import pytest
import logging
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.getcwd()))
from sift.learners import BARLightning


@pytest.fixture
def config_path():
    return "../sift/conf/bar_config.yaml"


@pytest.fixture
def config(config_path):
    return OmegaConf.load(config_path)


def test_bar(config):
    logging.info(config.data.voc)
    _ = BARLightning(config)
