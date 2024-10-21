import numpy as np
import os
import pytest
import parakeet.config
import parakeet.sample
import parakeet.sample.motion
import shutil
from math import sqrt


@pytest.fixture
def atom_data_4v5d():
    # Get the filename of the 4v5d.cif file
    filename = parakeet.data.get_path("4v5d.cif")

    # Get the atom data
    atoms = parakeet.sample.AtomData.from_gemmi_file(filename)
    atoms.data = parakeet.sample.recentre(atoms.data)
    return atoms


# def test_get_atom_sigma_sq():
#     pass

# def test_get_atom_sigma():
#     pass


def test_translate(atom_data_4v5d):
    data = atom_data_4v5d.data
    coords = data[["x", "y", "z"]].to_numpy()
    x00 = coords.min(axis=0)
    x01 = coords.max(axis=0)
    data = parakeet.sample.translate(atom_data_4v5d.data, (1, 2, 3))
    coords = data[["x", "y", "z"]].to_numpy()
    x10 = coords.min(axis=0)
    x11 = coords.max(axis=0)
    np.testing.assert_allclose(x10, x00 + (1, 2, 3))
    np.testing.assert_allclose(x11, x01 + (1, 2, 3))

    # Check dtypes
    for name, dtype in parakeet.sample.AtomData.column_data.items():
        assert data[name].dtype == dtype


def test_recentre(atom_data_4v5d):
    data = parakeet.sample.recentre(atom_data_4v5d.data)
    coords = data[["x", "y", "z"]]
    xm = coords.mean()
    np.testing.assert_allclose(xm, (0, 0, 0), atol=1e-5)

    # Check dtypes
    for name, dtype in parakeet.sample.AtomData.column_data.items():
        assert data[name].dtype == dtype


def test_number_of_water_molecules():
    n = parakeet.sample.number_of_water_molecules(1000**3)
    assert n == 31422283


def test_random_uniform_rotation():
    rotations = parakeet.sample.random_uniform_rotation(size=10)
    assert rotations.shape == (10, 3)


# def test_distribute_boxes_uniformly():

#     positions = parakeet.sample.distribute_boxes_uniformly(
#         ((0, 0, 0), (1000, 1000, 1000)), [(100, 100, 100), (200, 200, 200)]
#     )

#     assert len(positions) == 2


def test_shape_bounding_box():
    b1 = parakeet.sample.shape_bounding_box(
        (0, 0, 0), {"type": "cube", "cube": {"length": 1}}
    )
    assert pytest.approx(b1[0]) == (-0.5, -0.5, -0.5)
    assert pytest.approx(b1[1]) == (0.5, 0.5, 0.5)

    b2 = parakeet.sample.shape_bounding_box(
        (0, 0, 0),
        {"type": "cuboid", "cuboid": {"length_x": 1, "length_y": 2, "length_z": 3}},
    )
    assert pytest.approx(b2[0]) == (-0.5, -1.0, -1.5)
    assert pytest.approx(b2[1]) == (0.5, 1.0, 1.5)

    b3 = parakeet.sample.shape_bounding_box(
        (0, 0, 0), {"type": "cylinder", "cylinder": {"length": 1, "radius": 2}}
    )
    assert pytest.approx(b3[0]) == (-2.0, -0.5, -2.0)
    assert pytest.approx(b3[1]) == (2.0, 0.5, 2.0)


def test_shape_enclosed_box():
    b1 = parakeet.sample.shape_enclosed_box(
        (0, 0, 0), {"type": "cube", "cube": {"length": 1}}
    )
    assert pytest.approx(b1[0]) == (-0.5, -0.5, -0.5)
    assert pytest.approx(b1[1]) == (0.5, 0.5, 0.5)

    b2 = parakeet.sample.shape_enclosed_box(
        (0, 0, 0),
        {"type": "cuboid", "cuboid": {"length_x": 1, "length_y": 2, "length_z": 3}},
    )
    assert pytest.approx(b2[0]) == (-0.5, -1.0, -1.5)
    assert pytest.approx(b2[1]) == (0.5, 1.0, 1.5)

    b3 = parakeet.sample.shape_enclosed_box(
        (0, 0, 0), {"type": "cylinder", "cylinder": {"length": 1, "radius": 2}}
    )
    assert pytest.approx(b3[0]) == (-sqrt(2.0), -0.5, -sqrt(2.0))
    assert pytest.approx(b3[1]) == (sqrt(2.0), 0.5, sqrt(2.0))


def test_is_shape_inside_box():
    assert (
        parakeet.sample.is_shape_inside_box(
            (10, 10, 10), (0, 0, 0), {"type": "cube", "cube": {"length": 1}}
        )
        == False
    )

    assert (
        parakeet.sample.is_shape_inside_box(
            (10, 10, 10), (5, 5, 5), {"type": "cube", "cube": {"length": 1}}
        )
        == True
    )


def test_is_box_inside_shape():
    assert (
        parakeet.sample.is_box_inside_shape(
            ((0, 0, 0), (0.1, 0.1, 0.1)),
            (0, 0, 0),
            {"type": "cube", "cube": {"length": 1}},
        )
        == True
    )

    assert (
        parakeet.sample.is_box_inside_shape(
            ((0, 0, 0), (10, 10, 10)),
            (5, 5, 5),
            {"type": "cube", "cube": {"length": 1}},
        )
        == False
    )

    assert (
        parakeet.sample.is_box_inside_shape(
            ((0, 0, 0), (0.1, 0.1, 0.1)),
            (0, 0, 0),
            {"type": "cuboid", "cuboid": {"length_x": 1, "length_y": 2, "length_z": 3}},
        )
        == True
    )

    assert (
        parakeet.sample.is_box_inside_shape(
            ((0, 0, 0), (10, 10, 10)),
            (5, 5, 5),
            {"type": "cuboid", "cuboid": {"length_x": 1, "length_y": 2, "length_z": 3}},
        )
        == False
    )

    assert (
        parakeet.sample.is_box_inside_shape(
            ((0, 0, 0), (0.99 / sqrt(2), 0.3, 0.99 / sqrt(2))),
            (0, 0, 0),
            {"type": "cylinder", "cylinder": {"length": 1, "radius": 1}},
        )
        == True
    )

    assert (
        parakeet.sample.is_box_inside_shape(
            ((0, 0, 0), (10, 10, 10)),
            (0, 0, 0),
            {"type": "cylinder", "cylinder": {"length": 1, "radius": 1}},
        )
        == False
    )


def test_AtomData(atom_data_4v5d):
    # Check rotate doesn't modify types
    atom_data_4v5d.rotate((0, 0, 1))
    for name, dtype in parakeet.sample.AtomData.column_data.items():
        assert atom_data_4v5d.data[name].dtype == dtype

    # Check translate keeps types
    atom_data_4v5d.translate((0, 0, 1))
    for name, dtype in parakeet.sample.AtomData.column_data.items():
        assert atom_data_4v5d.data[name].dtype == dtype


def test_SampleHDF5Adapter(tmp_path, atom_data_4v5d):
    # Get handle
    handle = parakeet.sample.SampleHDF5Adapter(
        os.path.join(tmp_path, "test_SampleHDF5Adapter.h5"), "w"
    )

    # Test sample
    sample = handle.sample
    bounding_box = ((1, 2, 3), (4, 5, 6))
    containing_box = ((2, 3, 4), (5, 6, 7))
    centre = (1, 2, 2)
    shape = {"type": "cylinder", "cylinder": {"length": 10, "radius": 5}}
    sample.bounding_box = bounding_box
    sample.containing_box = containing_box
    sample.centre = centre
    sample.shape = shape
    assert (sample.bounding_box == bounding_box).all()
    assert (sample.containing_box == containing_box).all()
    assert (sample.centre == centre).all()
    assert sample.shape["type"] == shape["type"]
    assert sample.shape["cylinder"]["radius"] == shape["cylinder"]["radius"]
    assert sample.shape["cylinder"]["length"] == shape["cylinder"]["length"]

    # Test molecules
    molecules = sample.molecules
    assert len(molecules) == 0
    my_molecule = molecules["my_molcule"]
    positions = [(1, 2, 3), (4, 5, 6)]
    orientations = [(5, 6, 7), (8, 9, 10)]
    my_molecule.atoms = atom_data_4v5d.data
    my_molecule.positions = positions
    my_molecule.orientations = orientations
    assert my_molecule.atoms.equals(atom_data_4v5d.data)
    assert (my_molecule.positions == np.array(positions)).all()
    assert (my_molecule.orientations == np.array(orientations)).all()
    assert len(molecules) == 1

    # Test atoms
    atoms = sample.atoms
    assert len(atoms) == 0
    assert atoms.number_of_atoms() == 0
    atoms_000 = atoms["X=0; Y=0; Z=0"]
    atoms_001 = atoms["X=0; Y=0; Z=1"]
    atoms_000.atoms = atom_data_4v5d.data
    atoms_000.extend(atom_data_4v5d.data)
    atoms_001.extend(atom_data_4v5d.data)
    assert len(atoms_000) == atom_data_4v5d.data.shape[0] * 2
    assert len(atoms_001) == atom_data_4v5d.data.shape[0]
    assert atoms_001.atoms.equals(atom_data_4v5d.data)
    assert len(atoms) == 2
    assert atoms.number_of_atoms() == 3 * atom_data_4v5d.data.shape[0]

    handle.close()


def test_Sample(tmp_path, atom_data_4v5d):
    sample = parakeet.sample.Sample(os.path.join(tmp_path, "test_Sample.h5"), mode="w")

    assert sample.atoms_dataset_name((1, 2, 3)) == "X=000001; Y=000002; Z=000003"
    a = list(sample.atoms_dataset_range((1, 2, 3), (3, 4, 5)))
    assert (a[0][0] == (0, 0, 0)).all()
    assert (a[0][1] == (sample.step, sample.step, sample.step)).all()

    x0 = atom_data_4v5d.data[["x", "y", "z"]].min() + 200
    x1 = atom_data_4v5d.data[["x", "y", "z"]].max() + 200

    sample.add_molecule(
        atom_data_4v5d,
        positions=[(200, 200, 200)],
        orientations=[(0, 0, 0)],
        name="4v5d",
    )
    sample.containing_box = (0, 0, 0), (400, 400, 400)
    sample.centre = (200, 200, 200)
    sample.shape = {"type": "cube", "cube": {"length": 400}}

    assert (sample.bounding_box == (x0, x1)).all()
    assert (sample.containing_box == ((0, 0, 0), (400, 400, 400))).all()
    assert sample.molecules == ["4v5d"]
    assert sample.number_of_molecular_models == 1
    assert sample.number_of_molecules == 1
    assert (sample.dimensions == (x1 - x0)).all()

    atoms, positions, orientations = sample.get_molecule("4v5d")

    for name, data in sample.iter_molecules():
        assert name == "4v5d"

    assert sample.number_of_atoms == atom_data_4v5d.data.shape[0]

    for atoms in sample.iter_atoms():
        pass

    atoms = sample.get_atoms_in_range((100, 100, 100), (300, 300, 300)).data
    assert atoms.shape[0] > 0
    coords = atoms[["x", "y", "z"]].to_numpy()
    assert ((coords >= (100, 100, 100)) & (coords < (300, 300, 300))).all()

    sample.del_atoms(
        parakeet.sample.AtomDeleter(atom_data_4v5d, position=(200, 200, 200))
    )
    atoms = sample.get_atoms_in_range((0, 0, 0), (400, 400, 400)).data
    assert atoms.shape[0] == 0

    sample.add_atoms(atom_data_4v5d)

    sample.info()
    sample.close()


def test_AtomSliceExtractor(tmp_path):
    config = {
        "box": (50, 50, 50),
        "centre": (25, 25, 25),
        "shape": {"type": "cube", "cube": {"length": 40}},
        "ice": {"generate": True, "density": 940},
    }

    sample = parakeet.sample.new(
        parakeet.config.Sample(**config),
        os.path.join(tmp_path, "test_AtomSliceExtractor.h5"),
    )

    extractor = parakeet.sample.AtomSliceExtractor(sample, 0, 0.1, (0, 0), (50, 50))

    num_atoms = 0
    for zslice in extractor:
        coords = zslice.atoms.data[["x", "y", "z"]].to_numpy()
        assert (zslice.x_max[0] - zslice.x_min[0]) == pytest.approx(50)
        assert (zslice.x_max[1] - zslice.x_min[1]) == pytest.approx(50)
        assert (zslice.x_max[2] - zslice.x_min[2]) == pytest.approx(10)
        assert ((coords >= zslice.x_min) & (coords < zslice.x_max)).all(axis=1).all()
        num_atoms += zslice.atoms.data.shape[0]

    assert num_atoms == sample.number_of_atoms


def test_AtomDeleter(tmp_path):
    config = {
        "box": (50, 50, 50),
        "centre": (25, 25, 25),
        "shape": {"type": "cube", "cube": {"length": 40}},
        "ice": {"generate": True, "density": 940},
    }

    sample = parakeet.sample.new(
        parakeet.config.Sample(**config),
        os.path.join(tmp_path, "test_AtomDeleter.h5"),
    )

    atoms = sample.get_atoms_in_range((0, 0, 0), (50, 50, 50))
    coords = atoms.data[["x", "y", "z"]].to_numpy()

    deleter = parakeet.sample.AtomDeleter(
        parakeet.sample.AtomData(data=atoms.data[((coords - (50, 50, 50)) ** 2) < 20])
    )

    atoms = deleter(atoms.data)
    coords = atoms[["x", "y", "z"]].to_numpy()
    assert ((coords - (50, 50, 50)) ** 2 >= 20).all()


def test_load(tmp_path):
    config = {
        "box": (50, 50, 50),
        "centre": (25, 25, 25),
        "shape": {"type": "cylinder", "cylinder": {"length": 40, "radius": 20}},
    }

    sample = parakeet.sample.new(
        parakeet.config.Sample(**config),
        os.path.join(tmp_path, "test_load.h5"),
    )

    del sample

    sample = parakeet.sample.load(os.path.join(tmp_path, "test_load.h5"))


def test_new(tmp_path):
    config = {
        "box": (50, 50, 50),
        "centre": (25, 25, 25),
        "shape": {"type": "cylinder", "cylinder": {"length": 40, "radius": 20}},
    }

    sample = parakeet.sample.new(
        parakeet.config.Sample(**config),
        os.path.join(tmp_path, "test_new1.h5"),
    )

    config = {
        "box": (50, 50, 50),
        "centre": (25, 25, 25),
        "shape": {"type": "cylinder", "cylinder": {"length": 40, "radius": 20}},
        "ice": {"generate": True, "density": 940},
    }

    sample = parakeet.sample.new(
        parakeet.config.Sample(**config),
        os.path.join(tmp_path, "test_new2.h5"),
    )

    atoms = sample.get_atoms_in_range((0, 0, 0), (50, 50, 50))

    coords = atoms.data[["x", "y", "z"]].to_numpy()
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    margin = 2
    assert len(x) > 0
    assert ((y >= (5 - margin)) & (y < (45 + margin))).all()
    assert (((x - 25) ** 2 + (z - 25) ** 2) <= (20 + margin) ** 2).all()


def test_add_molecules(tmp_path):
    config = {
        "box": (1000, 1000, 1000),
        "centre": (500, 500, 500),
        "shape": {"type": "cube", "cube": {"length": 1000}},
    }

    sample = parakeet.sample.new(
        parakeet.config.Sample(**config),
        os.path.join(tmp_path, "test_add_molecules.h5"),
    )

    sample.close()

    sample = parakeet.sample.add_molecules(
        parakeet.config.Sample(
            **{"molecules": {"pdb": [{"id": "4v5d", "instances": 1}]}}
        ),
        parakeet.sample.load(os.path.join(tmp_path, "test_add_molecules.h5"), "r+"),
    )

    assert sample.number_of_molecules == 1
    assert sample.number_of_molecular_models == 1

    sample = parakeet.sample.add_molecules(
        parakeet.config.Sample(
            **{"molecules": {"pdb": [{"id": "4v1w", "instances": 10}]}}
        ),
        parakeet.sample.load(os.path.join(tmp_path, "test_add_molecules.h5"), "r+"),
    )

    assert sample.number_of_molecules == 2
    assert sample.number_of_molecular_models == 11

    sample.close()

    sample = parakeet.sample.new(
        parakeet.config.Sample(
            **{
                "box": (4000, 4000, 4000),
                "centre": (2000, 2000, 2000),
                "shape": {"type": "cylinder", "cylinder": {"length": 40, "radius": 20}},
                "ice": {"generate": True, "density": 940},
            }
        ),
        os.path.join(tmp_path, "test_add_molecules.h5"),
    )

    sample.shape = {"type": "cylinder", "cylinder": {"length": 4000, "radius": 2000}}
    sample.close()

    sample = parakeet.sample.add_molecules(
        parakeet.config.Sample(
            **{"molecules": {"pdb": [{"id": "4v5d", "instances": 1}]}}
        ),
        parakeet.sample.load(os.path.join(tmp_path, "test_add_molecules.h5"), "r+"),
    )

    assert sample.number_of_molecules == 1
    assert sample.number_of_molecular_models == 1

    # Test adding multiple
    sample = parakeet.sample.add_molecules(
        parakeet.config.Sample(
            **{"molecules": {"pdb": [{"id": "4v5d", "instances": 2}]}}
        ),
        parakeet.sample.load(os.path.join(tmp_path, "test_add_molecules.h5"), "r+"),
    )

    assert sample.number_of_molecules == 1
    assert sample.number_of_molecular_models == 3


def test_sample_new_with_local(tmp_path):
    filename = os.path.join(tmp_path, "my.cif")

    src = parakeet.data.get_4v1w()
    shutil.copyfile(src, filename)

    config = {
        "box": (50, 50, 50),
        "centre": (25, 25, 25),
        "shape": {"type": "cylinder", "cylinder": {"length": 40, "radius": 20}},
        "molecules": {
            "local": [
                {
                    "filename": filename,
                    "instances": 1,
                }
            ]
        },
    }

    sample = parakeet.sample.new(
        parakeet.config.Sample(**config),
        os.path.join(tmp_path, "test_new2.h5"),
    )


def test_sample_with_motion(tmp_path):

    filename = os.path.join(tmp_path, "my.cif")

    src = parakeet.data.get_4v1w()
    shutil.copyfile(src, filename)

    config = {
        "box": (400, 400, 400),
        "centre": (200, 200, 200),
        "shape": {
            "type": "cuboid",
            "cuboid": {"length_x": 400, "length_y": 400, "length_z": 400},
        },
        "molecules": {
            "local": [
                {
                    "filename": filename,
                    "instances": 10,
                }
            ]
        },
        "motion": {
            "global_drift": (1, 2),
            "interaction_range": 300,
            "velocity": 1,
            "noise_magnitude": 0,
        },
    }

    scan_config = {
        "mode": "tilt_series",
        "start_angle": 0,
        "step_angle": 1,
        "num_images": 2,
        "num_fractions": 10,
    }

    sample = parakeet.sample.new(
        parakeet.config.Sample(**config),
        os.path.join(tmp_path, "test_new3.h5"),
    )

    sample = parakeet.sample.add_molecules(parakeet.config.Sample(**config), sample)

    atoms = sample.get_atoms()

    position = []

    groups = list(set(atoms.data["group"]))

    for group in groups:
        select = atoms.data["group"] == group
        xc = np.mean(atoms.data[select]["x"])
        yc = np.mean(atoms.data[select]["y"])
        zc = np.mean(atoms.data[select]["z"])
        position.append((xc, yc, zc))

    assert len(position) == 10

    scan = parakeet.scan.new(**scan_config)

    position = np.array(position)
    direction = np.random.uniform(-np.pi, np.pi, size=position.shape[0])

    global_drift = config["motion"]["global_drift"]
    interaction_range = config["motion"]["interaction_range"]
    velocity = config["motion"]["velocity"]
    noise_magnitude = np.radians(config["motion"]["noise_magnitude"])

    for image_number, fraction_number, angle in zip(
        scan.image_number, scan.fraction_number, scan.angles
    ):
        position, direction = (
            parakeet.sample.motion.update_particle_position_and_direction(
                position,
                direction,
                global_drift,
                interaction_range,
                velocity,
                noise_magnitude,
            )
        )
        print(
            image_number, fraction_number, angle, position[0], np.degrees(direction[0])
        )
        assert len(position) == 10
        assert len(direction) == 10
