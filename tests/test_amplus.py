import pytest
import os.path
import yaml
import parakeet.config
import parakeet.command_line.sample
import parakeet.command_line.simulate
import parakeet.command_line.analyse


@pytest.fixture(scope="session")
def config_path(tmpdir_factory):
    directory = tmpdir_factory.mktemp("proc")

    config_dict = {
        "microscope": {
            "detector": {"nx": 250, "ny": 250, "pixel_size": 2},
            "beam": {
                "electrons_per_angstrom": 1000,
            },
        },
        "sample": {
            "molecules": {"pdb": [{"id": "4v1w", "instances": 2}]},
            "shape": {
                "type": "cube",
                "cube": {"length": 500},
            },
            "centre": (250, 250, 250),
            "box": (500, 500, 500),
        },
        "scan": {
            "mode": "tilt_series",
            "num_images": 10,
            "start_angle": -90,
            "step_angle": 18,
        },
    }

    config = parakeet.config.load(config_dict)
    config_path = os.path.abspath(os.path.join(directory, "config.yaml"))
    yaml.safe_dump(config.dict(), open(config_path, "w"))
    return directory


def test_sample_new(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    assert os.path.exists(config)
    parakeet.command_line.sample.new(["-c", config, "-s", sample])
    assert os.path.exists(sample)


def test_sample_add_molecules(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    assert os.path.exists(config)
    assert os.path.exists(sample)
    parakeet.command_line.sample.add_molecules(["-c", config, "-s", sample])


def test_simulate_exit_wave(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    exit_wave = os.path.abspath(os.path.join(config_path, "exit_wave.h5"))
    assert os.path.exists(config)
    assert os.path.exists(sample)
    parakeet.command_line.simulate.exit_wave(
        ["-c", config, "-s", sample, "-e", exit_wave]
    )
    assert os.path.exists(exit_wave)


def test_simulate_optics(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    exit_wave = os.path.abspath(os.path.join(config_path, "exit_wave.h5"))
    optics = os.path.abspath(os.path.join(config_path, "optics.h5"))
    assert os.path.exists(config)
    assert os.path.exists(exit_wave)
    parakeet.command_line.simulate.optics(["-c", config, "-e", exit_wave, "-o", optics])
    assert os.path.exists(optics)


def test_simulate_image(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    optics = os.path.abspath(os.path.join(config_path, "optics.h5"))
    image = os.path.abspath(os.path.join(config_path, "image.mrc"))
    assert os.path.exists(config)
    assert os.path.exists(optics)
    parakeet.command_line.simulate.image(["-c", config, "-o", optics, "-i", image])
    assert os.path.exists(image)


def test_analyse_correct(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    image = os.path.abspath(os.path.join(config_path, "image.mrc"))
    corrected = os.path.abspath(os.path.join(config_path, "corrected.mrc"))
    assert os.path.exists(config)
    assert os.path.exists(image)
    parakeet.command_line.analyse.correct(
        ["-c", config, "-i", image, "-cr", corrected, "-d", "cpu"]
    )
    assert os.path.exists(corrected)


def test_analyse_reconstruct(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    image = os.path.abspath(os.path.join(config_path, "image.mrc"))
    rec = os.path.abspath(os.path.join(config_path, "rec.mrc"))
    assert os.path.exists(config)
    assert os.path.exists(image)
    parakeet.command_line.analyse.reconstruct(
        ["-c", config, "-i", image, "-r", rec, "-d", "cpu"]
    )
    assert os.path.exists(rec)


def test_analyse_average_particles(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    rec = os.path.abspath(os.path.join(config_path, "rec.mrc"))
    half1 = os.path.abspath(os.path.join(config_path, "half1.mrc"))
    half2 = os.path.abspath(os.path.join(config_path, "half2.mrc"))
    assert os.path.exists(config)
    assert os.path.exists(sample)
    assert os.path.exists(rec)
    parakeet.command_line.analyse.average_particles(
        ["-c", config, "-s", sample, "-r", rec, "-h1", half1, "-h2", half2]
    )
    assert os.path.exists(half1)
    assert os.path.exists(half2)


def test_export(config_path):

    exit_wave1 = os.path.abspath(os.path.join(config_path, "exit_wave.h5"))
    exit_wave2 = os.path.abspath(os.path.join(config_path, "exit_wave2.mrc"))
    exit_wave3 = os.path.abspath(os.path.join(config_path, "exit_wave3.h5"))
    optics1 = os.path.abspath(os.path.join(config_path, "optics.h5"))
    optics2 = os.path.abspath(os.path.join(config_path, "optics2.mrc"))
    optics3 = os.path.abspath(os.path.join(config_path, "optics3.h5"))
    image1 = os.path.abspath(os.path.join(config_path, "image.mrc"))
    image2 = os.path.abspath(os.path.join(config_path, "image2.h5"))
    image3 = os.path.abspath(os.path.join(config_path, "image3.mrc"))
    corrected1 = os.path.abspath(os.path.join(config_path, "corrected.mrc"))

    assert os.path.exists(exit_wave1)
    assert os.path.exists(optics1)
    assert os.path.exists(image1)
    assert os.path.exists(corrected1)

    parakeet.command_line.export([exit_wave1, "-o", exit_wave2])
    parakeet.command_line.export([exit_wave2, "-o", exit_wave3])
    parakeet.command_line.export([optics1, "-o", optics2])
    parakeet.command_line.export([optics2, "-o", optics3])
    parakeet.command_line.export([image1, "-o", image2])
    parakeet.command_line.export([image2, "-o", image3])

    assert os.path.exists(exit_wave2)
    assert os.path.exists(exit_wave3)
    assert os.path.exists(optics2)
    assert os.path.exists(optics3)
    assert os.path.exists(image2)
    assert os.path.exists(image3)
