import glob
import pytest
import os
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
                "electrons_per_angstrom": 10000,
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
        "simulation": {
            "ice": True,
        },
    }

    config = parakeet.config.load(config_dict)
    config_path = os.path.abspath(os.path.join(directory, "config.yaml"))
    yaml.safe_dump(config.dict(), open(config_path, "w"))

    config_dict["sample"]["motion"] = {
        "interaction_range": 100,
        "velocity": 1,
        "noise_magnitude": 1,
    }

    config_dict["scan"] = {
        "mode": "tilt_series",
        "num_images": 2,
        "num_fractions": 3,
        "start_angle": 0,
        "step_angle": 2,
    }
    config = parakeet.config.load(config_dict)
    config_path = os.path.abspath(os.path.join(directory, "config_with_motion.yaml"))
    yaml.safe_dump(config.dict(), open(config_path, "w"))

    return directory


def test_config_new(config_path):
    config = os.path.abspath(os.path.join(config_path, "config-new.yaml"))
    parakeet.command_line.config.new(["-c", config])
    assert os.path.exists(config)


def test_config_edit(config_path):
    config_in = os.path.abspath(os.path.join(config_path, "config.yaml"))
    config_out = os.path.abspath(os.path.join(config_path, "config-edit.yaml"))
    config_str = "microscope:\n" "  beam:\n" "    energy: 200\n"
    assert os.path.exists(config_in)
    parakeet.command_line.config.edit(
        ["-i", config_in, "-o", config_out, "-s", config_str]
    )
    assert os.path.exists(config_out)
    config = parakeet.config.load(config_out)
    assert config.microscope.beam.energy == 200


def test_config_show(config_path):
    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    assert os.path.exists(config)
    parakeet.command_line.config.show(["-c", config])


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


def test_sample_mill(config_path):
    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    assert os.path.exists(config)
    assert os.path.exists(sample)
    parakeet.command_line.sample.mill(["-c", config, "-s", sample])


def test_sample_sputter(config_path):
    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    assert os.path.exists(config)
    assert os.path.exists(sample)
    parakeet.command_line.sample.sputter(["-c", config, "-s", sample])


def test_sample_show(config_path):
    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    assert os.path.exists(config)
    assert os.path.exists(sample)
    parakeet.command_line.sample.show(["-s", sample])


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


def test_simulate_exit_wave_with_motion(config_path):
    config = os.path.abspath(os.path.join(config_path, "config_with_motion.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    exit_wave = os.path.abspath(os.path.join(config_path, "exit_wave_with_motion.h5"))
    assert os.path.exists(config)
    assert os.path.exists(sample)
    parakeet.command_line.simulate.exit_wave(
        ["-c", config, "-s", sample, "-e", exit_wave]
    )
    assert os.path.exists(exit_wave)


def test_simulate_cbed(config_path):
    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    cbed = os.path.abspath(os.path.join(config_path, "cbed.h5"))
    assert os.path.exists(config)
    assert os.path.exists(sample)
    parakeet.command_line.simulate.cbed(["-c", config, "-s", sample, "-i", cbed])
    assert os.path.exists(cbed)


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


def test_simulate_ctf(config_path):
    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    ctf = os.path.abspath(os.path.join(config_path, "ctf.h5"))
    assert os.path.exists(config)
    parakeet.command_line.simulate.ctf(["-c", config, "-o", ctf])
    assert os.path.exists(ctf)


def test_simulate_potential(config_path):
    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    potential = os.path.abspath(os.path.join(config_path, "potential"))
    assert os.path.exists(config)
    assert os.path.exists(sample)
    parakeet.command_line.simulate.potential(
        ["-c", config, "-s", sample, "-p", potential]
    )
    filenames = glob.glob("%s*.mrc" % potential)
    assert len(filenames) > 0


def test_simulate_simple(config_path):
    pass


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
    rec_cpu = os.path.abspath(os.path.join(config_path, "rec.mrc"))
    # rec_gpu = os.path.abspath(os.path.join(config_path, "rec_gpu.mrc"))
    assert os.path.exists(config)
    assert os.path.exists(image)
    parakeet.command_line.analyse.reconstruct(
        ["-c", config, "-i", image, "-r", rec_cpu, "-d", "cpu"]
    )
    # parakeet.command_line.analyse.reconstruct(
    #     ["-c", config, "-i", image, "-r", rec_gpu, "-d", "gpu"]
    # )
    assert os.path.exists(rec_cpu)
    # assert os.path.exists(rec_gpu)

    # FIXME TEST GIVES THE SAME RESULT
    # d1 = mrcfile.open(rec_cpu).data
    # d2 = mrcfile.open(rec_gpu).data
    # assert np.all(np.isclose(d1, d2))


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


def test_analyse_average_all_particles(config_path):
    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    rec = os.path.abspath(os.path.join(config_path, "rec.mrc"))
    average = os.path.abspath(os.path.join(config_path, "average.mrc"))
    assert os.path.exists(config)
    assert os.path.exists(sample)
    assert os.path.exists(rec)
    parakeet.command_line.analyse.average_all_particles(
        ["-c", config, "-s", sample, "-r", rec, "-avm", average]
    )
    assert os.path.exists(average)


def test_analyse_extract(config_path):
    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample.h5"))
    rec = os.path.abspath(os.path.join(config_path, "rec.mrc"))
    particles = os.path.abspath(os.path.join(config_path, "particles.h5"))
    assert os.path.exists(config)
    assert os.path.exists(sample)
    assert os.path.exists(rec)
    parakeet.command_line.analyse.extract(
        ["-c", config, "-s", sample, "-r", rec, "-pm", particles]
    )
    assert os.path.exists(particles)


def test_analyse_refine(config_path):
    pass


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


def test_pdb_read(config_path):
    pdb = parakeet.data.get_pdb("4v1w")
    assert os.path.exists(pdb)
    parakeet.command_line.pdb.read([pdb])


def test_pdb_get1(config_path):
    directory = os.path.abspath(config_path)
    parakeet.command_line.pdb.get(["4v1w", "-d", directory])
    assert os.path.exists(os.path.join(directory, "4v1w.cif"))


def test_pdb_get2(config_path):
    if os.getenv("CI"):
        pytest.skip("Doesn't work on github workflow")
        return
    directory = os.path.abspath(config_path)
    parakeet.command_line.pdb.get(["1uad", "-d", directory])
    assert os.path.exists(os.path.join(directory, "1uad.cif"))


def test_run(config_path):
    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample-run.h5"))
    exit_wave = os.path.abspath(os.path.join(config_path, "exit_wave-run.h5"))
    optics = os.path.abspath(os.path.join(config_path, "optics-run.h5"))
    image = os.path.abspath(os.path.join(config_path, "image-run.mrc"))
    assert os.path.exists(config)
    parakeet.command_line.run(
        ["-c", config, "-s", sample, "-e", exit_wave, "-o", optics, "-i", image]
    )
    assert os.path.exists(sample)
    assert os.path.exists(exit_wave)
    assert os.path.exists(optics)
    assert os.path.exists(image)


def test_main(config_path):
    config = os.path.abspath(os.path.join(config_path, "config.yaml"))
    sample = os.path.abspath(os.path.join(config_path, "sample-main.h5"))
    exit_wave = os.path.abspath(os.path.join(config_path, "exit_wave-main.h5"))
    optics = os.path.abspath(os.path.join(config_path, "optics-main.h5"))
    image = os.path.abspath(os.path.join(config_path, "image-main.mrc"))
    assert os.path.exists(config)
    parakeet.command_line.main(
        ["run", "-c", config, "-s", sample, "-e", exit_wave, "-o", optics, "-i", image]
    )
    assert os.path.exists(sample)
    assert os.path.exists(exit_wave)
    assert os.path.exists(optics)
    assert os.path.exists(image)
