import os
import yaml
import parakeet.config


def test_temp_directory():

    assert parakeet.config.temp_directory() == "_parakeet"


def test_default():

    config = parakeet.config.default()


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
        "C": {"D": {"E": False}, "F": [4, 5, 6], "G": [{"H": 0.3}, {"H": 0.4}]},
        "I": "unchanged",
        "J": "also unchanged",
        "K": ["Spam", "Eggs"],
    }

    assert parakeet.config.deepmerge(a, b) == expected
    assert parakeet.config.deepmerge(a, {}) == a
    assert parakeet.config.deepmerge({}, b) == b


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

    assert parakeet.config.difference(master, config) == expected


def test_load(tmp_path):

    config = parakeet.config.load()

    filename = os.path.join(tmp_path, "tmp.yaml")
    with open(filename, "w") as outfile:
        yaml.dump(config, outfile)

    config = parakeet.config.load(filename)

    expected = {
        "sample": {
            "box": [4000, 4000, 4000],
            "centre": [2000, 2000, 2000],
            "shape": {"margin": [0, 0, 0]},
            "coords": {
                "orientation": [0, 0, 0],
            },
            "molecules": {
                "local": [],
                "pdb": [],
            },
        },
        "scan": {"axis": [0, 1, 0]},
        "microscope": {"detector": {"origin": [0, 0]}},
    }
    assert parakeet.config.difference(config, parakeet.config.default()) == expected


def test_show():

    parakeet.config.show({})
