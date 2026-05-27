import numpy as np
import pandas as pd
import pytest

from src import config


def test_config_class_names_are_alphabetical():
    assert config.NUM_CLASSES == 15
    assert config.CLASS_NAMES == sorted(config.CLASS_NAMES)
