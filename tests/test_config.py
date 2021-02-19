import os
import yaml
import amplus.config


def test_temp_directory():

    assert amplus.config.temp_directory() == "_amplus"


def test_default():

    config = amplus.config.default()


def test_deepmerge():

    a = {
        "A": 10,
        "B": "Hello World",
        "C": {"D": {"E": True}, "F": [1, 2, 3], "G": [{"H": 0.1}, {"H": 0.2}]},
        "I": "unchanged",
    }

    b = {
        "A": 20,
        "C": {"D": {"E": False}, "F": [4, 5, 6], "G": [{"H": 0.3}, {"H": 0.4}]},
        "J": "also unchanged",
        "K": ["Spam", "Eggs"],
    }

    expected = {
        "A": 20,
        "B": "Hello World",
        "C": {
            "D": {"E": False},
            "F": [1, 2, 3, 4, 5, 6],
            "G": [{"H": 0.1}, {"H": 0.2}, {"H": 0.3}, {"H": 0.4}],
        },
        "I": "unchanged",
        "J": "also unchanged",
        "K": ["Spam", "Eggs"],
    }

    assert amplus.config.deepmerge(a, b) == expected
    assert amplus.config.deepmerge(a, {}) == a
    assert amplus.config.deepmerge({}, b) == b


def test_difference():

    master = {
        "A": 10,
        "B": "Hello World",
        "C": {"D": {"E": True}, "F": [1, 2, 3], "G": [{"H": 0.1}, {"H": 0.2}]},
        "I": "a string",
        "J": "another string",
        "K": ["Spam", "Eggs"],
    }

    config = {
        "A": 10,
        "C": {"D": {"E": False}, "F": [4, 5, 6], "G": [{"H": 0.3}, {"H": 0.4}]},
        "I": "modified",
        "K": ["Beans"],
    }

    expected = {
        "C": {"D": {"E": False}, "F": [4, 5, 6], "G": [{"H": 0.3}, {"H": 0.4}]},
        "I": "modified",
        "K": ["Beans"],
    }

    assert amplus.config.difference(master, config) == expected


def test_load(tmp_path):

    config = amplus.config.load()

    filename = os.path.join(tmp_path, "tmp.yaml")
    with open(filename, "w") as outfile:
        yaml.dump(config, outfile)

    config = amplus.config.load(filename)

    expected = {"scan": {"axis": [1, 0, 0]}}
    assert amplus.config.difference(config, amplus.config.default()) == expected


def test_show():

    amplus.config.show({})
