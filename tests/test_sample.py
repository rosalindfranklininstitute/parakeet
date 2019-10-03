import numpy
import os
import pytest
import elfantasma.sample
from math import pi


@pytest.fixture
def sample_4v5d():
    return elfantasma.sample.new("4v5d")


def test_structure(sample_4v5d):

    # Get the sample atom data
    atom_data = sample_4v5d.structures[0].atom_data

    # Create a new structure
    structure = elfantasma.sample.Structure(atom_data)
    assert len(structure.positions) == 0
    assert len(structure.rotations) == 0

    # Check some characteristics
    assert structure.num_models == 0
    assert structure.num_atoms == 0
    assert numpy.allclose(structure.bounding_box[0], (0, 0, 0))
    assert numpy.allclose(structure.bounding_box[1], (0, 0, 0))

    # Add a model
    structure.append([[0, 0, 0]], [[0, 0, 0]])

    # Check some characteristics
    assert structure.num_models == 1
    assert structure.num_atoms == 296042
    assert numpy.allclose(structure.bounding_box[0], (-133.762, -195.798, -190.784))
    assert numpy.allclose(structure.bounding_box[1], (133.762, 195.798, 190.784))

    # Add a model
    structure.append([[10, 10, 10]], [[0, 0, 0]])

    # Check some characteristics
    assert structure.num_models == 2
    assert structure.num_atoms == 296042 * 2
    assert numpy.allclose(structure.bounding_box[0], (-133.762, -195.798, -190.784))
    assert numpy.allclose(structure.bounding_box[1], (143.762, 205.798, 200.784))

    # Add a model
    structure.append([[200, 200, 200]], [[pi / 2, pi / 2, pi / 2]])

    # Check some characteristics
    assert structure.num_models == 3
    assert structure.num_atoms == 296042 * 3
    assert numpy.allclose(structure.bounding_box[0], (-133.762, -195.798, -190.784))
    assert numpy.allclose(
        structure.bounding_box[1], (399.42072897, 342.17062303, 402.50910872)
    )

    # Translate the structure
    structure.translate((10, 20, 30))

    # Check the positions
    assert numpy.allclose(
        structure.positions, [(10, 20, 30), (20, 30, 40), (210, 220, 230)]
    )

    # Rotate the structure
    structure.rotate((pi / 2, 0, 0))

    # Check the rotations
    assert numpy.allclose(
        structure.rotations,
        [(pi / 2, 0, 0), (pi / 2, 0, 0), (-1.48800919, 0, -2.17230367)],
    )

    # Get the atoms
    assert len(list(structure.spec_atoms)) == structure.num_atoms


def test_sample(tmp_path, sample_4v5d):

    sample = sample_4v5d

    assert sample.num_models == 1
    assert sample.num_atoms == 296042
    assert numpy.allclose(sample.sample_bbox[0], (133.762, 195.798, 190.784))
    assert numpy.allclose(sample.sample_bbox[1], (401.286, 587.394, 572.352))

    sample.resize()
    sample.recentre()
    sample.update()

    assert numpy.allclose(sample.sample_bbox[0], (1, 1, 1))
    assert numpy.allclose(sample.sample_bbox[1], sample.box_size - 1)

    sample.validate()

    text = sample.info()
    assert text == (
        "Sample information:\n"
        "    # models:      1\n"
        "    # atoms:       296042\n"
        "    Min x:         1.00\n"
        "    Min y:         1.00\n"
        "    Min z:         1.00\n"
        "    Max x:         268.52\n"
        "    Max y:         392.60\n"
        "    Max z:         382.57\n"
        "    Sample size x: 267.52\n"
        "    Sample size y: 391.60\n"
        "    Sample size z: 381.57\n"
        "    Box size x:    269.52\n"
        "    Box size y:    393.60\n"
        "    Box size z:    383.57"
    )

    sample.structures[0].append([[10, 10, 10]], [[0, 0, 0]])
    sample.structures[0].append([[200, 200, 200]], [[pi / 2, pi / 2, pi / 2]])

    sample.update()
    sample.resize()
    sample.recentre()

    assert sample.num_models == 3
    assert sample.num_atoms == 296042 * 3
    assert numpy.allclose(sample.sample_bbox[0], (1, 1, 1))
    assert numpy.allclose(sample.sample_bbox[1], sample.box_size - 1)

    sample.translate((10, 10, 10))

    assert numpy.allclose(sample.sample_bbox[0], (11, 11, 11))
    assert numpy.allclose(sample.sample_bbox[1], sample.box_size - 1 + 10)

    sample.rotate((pi / 2, 0, 0))

    assert numpy.allclose(sample.sample_bbox[0], (11.0, 16.014, 5.986))
    assert numpy.allclose(sample.sample_bbox[1], (534.18272897, 584.38, 579.366))

    sample.update()
    sample.resize()
    sample.recentre()

    assert numpy.allclose(sample.sample_bbox[0], (1, 1, 1))
    assert numpy.allclose(sample.sample_bbox[1], sample.box_size - 1)
    assert len(list(sample.spec_atoms)) == 296042 * 3

    sample.append(sample.structures[0])

    assert sample.num_models == 6
    assert sample.num_atoms == 296042 * 6
    assert len(list(sample.spec_atoms)) == 296042 * 6

    sample.update()
    sample.resize()
    sample.recentre()
    sample.validate()

    subset = sample.select_atom_data_in_roi([(50, 50), (100, 100)])

    assert len(subset) < sample.num_atoms
    assert numpy.all(numpy.greater_equal(subset["x"], 50))
    assert numpy.all(numpy.greater_equal(subset["y"], 50))
    assert numpy.all(numpy.less_equal(subset["x"], 100))
    assert numpy.all(numpy.less_equal(subset["y"], 100))

    cif_path = os.path.join(tmp_path, "tmp.cif")
    pdb_path = os.path.join(tmp_path, "tmp.pdb")
    pkl_path = os.path.join(tmp_path, "tmp.pickle")

    sample.as_file(cif_path)
    sample.as_file(pdb_path)
    sample.as_file(pkl_path)

    assert os.path.exists(cif_path)
    assert os.path.exists(pdb_path)
    assert os.path.exists(pkl_path)

    # sample2 = elfantasma.sample.load(cif_path) # Bug in gemmi
    sample2 = elfantasma.sample.load(pdb_path)
    sample2 = elfantasma.sample.load(pkl_path)


def test_create_ribosomes_in_lamella_sample():

    sample = elfantasma.sample.create_ribosomes_in_lamella_sample(4000, 4000, 500)

    assert sample.box_size[0] == 4000
    assert sample.box_size[1] == 4000
    assert sample.box_size[2] == 500

    sample.validate()


def test_create_ribosomes_in_cylinder_sample():

    sample = elfantasma.sample.create_ribosomes_in_cylinder_sample(1500, 500, 10000)

    assert sample.box_size[0] == 10000
    assert sample.box_size[1] == 4000
    assert sample.box_size[2] == 4000

    sample.validate()
