import os
import yaml
import elfantasma.config


def test_temp_directory():

    assert elfantasma.config.temp_directory() == "_elfantasma"


def test_default():

    config = elfantasma.config.default()


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

    assert elfantasma.config.deepmerge(a, b) == expected
    assert elfantasma.config.deepmerge(a, {}) == a
    assert elfantasma.config.deepmerge({}, b) == b


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

    assert elfantasma.config.difference(master, config) == expected


def test_load(tmp_path):

    config = elfantasma.config.load()

    filename = os.path.join(tmp_path, "tmp.yaml")
    with open(filename, "w") as outfile:
        yaml.dump(config, outfile)

    config = elfantasma.config.load(filename)

    expected = {"scan": {"axis": [1, 0, 0]}}
    assert elfantasma.config.difference(config, elfantasma.config.default()) == expected


def test_show():

    elfantasma.config.show({})
