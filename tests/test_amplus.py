import pytest
import os.path
import yaml
import amplus.config
import amplus.command_line.sample
import amplus.command_line.simulate
import amplus.command_line.analyse


@pytest.fixture(scope="session")
def config_path(tmpdir_factory):
    directory = fn = tmpdir_factory.mktemp("proc")
    config = amplus.config.load()
    config["microscope"]["detector"]["nx"] = 600
    config["microscope"]["detector"]["ny"] = 600
    config["microscope"]["beam"]["electrons_per_angstrom"] = 1000
    config["sample"]["molecules"]["4v1w"] = 2
    config["sample"]["shape"]["type"] = "cube"
    config["sample"]["shape"]["cube"]["length"] = 600
    config["sample"]["centre"] = (300, 300, 300)
    config["sample"]["box"] = (600, 600, 600)
    config["scan"]["mode"] = "tilt_series"
    config["scan"]["num_images"] = 10
    config["scan"]["start_angle"] = -90
    config["scan"]["step_angle"] = 18
    config_path = os.path.abspath(os.path.join(directory, "config.yaml"))
    yaml.safe_dump(config, open(config_path, "w"))
    return directory


def test_sample_new(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    amplus.command_line.sample.new(["-c", config, "-s", sample])


def test_sample_add_molecules(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    amplus.command_line.sample.add_molecules(["-c", config, "-s", sample])


def test_simulate_exit_wave(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    exit_wave = os.path.abspath(os.path.join(config_path, "exit_wave.h5"))
    amplus.command_line.simulate.exit_wave(
        ["-c", config, "-s", sample, "-e", exit_wave]
    )


def test_simulate_optics(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    exit_wave = os.path.abspath(os.path.join(config_path, "exit_wave.h5"))
    optics = os.path.abspath(os.path.join(config_path, "optics.h5"))
    amplus.command_line.simulate.optics(["-c", config, "-e", exit_wave, "-o", optics])


def test_simulate_image(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    optics = os.path.abspath(os.path.join(config_path, "optics.h5"))
    image = os.path.abspath(os.path.join(config_path, "image.mrc"))
    amplus.command_line.simulate.image(["-c", config, "-o", optics, "-i", image])


def test_analyse_reconstruct(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    image = os.path.abspath(os.path.join(config_path, "image.mrc"))
    rec = os.path.abspath(os.path.join(config_path, "rec.mrc"))
    amplus.command_line.analyse.reconstruct(["-c", config, "-i", image, "-r", rec])


def test_analyse_average_particles(config_path):

    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    rec = os.path.abspath(os.path.join(config_path, "rec.mrc"))
    half1 = os.path.abspath(os.path.join(config_path, "half1.mrc"))
    half2 = os.path.abspath(os.path.join(config_path, "half2.mrc"))
    amplus.command_line.analyse.average_particles(
        ["-c", config, "-s", sample, "-r", rec, "-h1", half1, "-h2", half2]
    )
