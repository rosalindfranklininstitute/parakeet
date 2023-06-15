import numpy as np
import os
import pytest
import parakeet.io


@pytest.fixture
def io_test_data():
    # The shape of the data
    shape = (10, 100, 100)

    # Generate the data
    data = np.random.randint(low=0, high=100, size=shape[0] * shape[1] * shape[2])
    data.shape = shape

    # Generate the angles
    angle = np.array([x for x in range(0, shape[0])])

    # Generate the positons
    position = np.array([[x, x, 0] for x in range(0, shape[0])])

    # Return the data
    return (data, angle, position)


def test_read_write_mrcfile(tmp_path, io_test_data):
    filename = os.path.join(tmp_path, "tmp.mrc")

    data, angle, position = io_test_data

    writer = parakeet.io.new(filename, shape=data.shape)
    for i in range(data.shape[0]):
        writer.data[i, :, :] = data[i, :, :]
        writer.header[i]["tilt_alpha"] = angle[i]
        writer.header[i]["shift_x"] = position[i][0]
        writer.header[i]["shift_y"] = position[i][1]
        writer.header[i]["stage_z"] = position[i][2]

    assert writer.shape == data.shape
    assert writer.is_mrcfile_writer == True
    assert writer.is_nexus_writer == False
    assert writer.is_image_writer == False

    # Test different ways of writing position
    writer.header.position[0, 0] = position[0][0]
    writer.header.position[0, 1] = position[0][1]
    writer.header.position[0, 2] = position[0][2]
    writer.header.position[2, :] = position[2]
    writer.header.position[3, 0:2] = position[3][0:2]
    writer.header.position[3, 2] = position[3][2]

    # Make sure stuff is written
    writer = None

    reader = parakeet.io.open(filename)
    assert reader.data.shape == (10, 100, 100)
    assert reader.header.size == 10
    assert np.all(np.equal(reader.header["tilt_alpha"], angle))
    assert np.all(np.equal(reader.header.position, position))


def test_write_nexus(tmp_path, io_test_data):
    filename = os.path.join(tmp_path, "tmp.h5")

    data, angle, position = io_test_data

    writer = parakeet.io.new(filename, shape=data.shape)
    for i in range(data.shape[0]):
        writer.data[i, :, :] = data[i, :, :]
        writer.header[i]["tilt_alpha"] = angle[i]
        writer.header[i]["shift_x"] = position[i][0]
        writer.header[i]["shift_y"] = position[i][1]
        writer.header[i]["stage_z"] = position[i][2]

    assert writer.shape == data.shape
    assert writer.is_mrcfile_writer == False
    assert writer.is_nexus_writer == True
    assert writer.is_image_writer == False

    # Test different ways of writing position
    writer.header.position[0, 0] = position[0][0]
    writer.header.position[0, 1] = position[0][1]
    writer.header.position[0, 2] = position[0][2]
    writer.header.position[2, :] = position[2]
    writer.header.position[3, 0:2] = position[3][0:2]
    writer.header.position[3, 2] = position[3][2]

    # Make sure stuff is written
    writer = None

    reader = parakeet.io.open(filename)
    assert reader.data.shape == (10, 100, 100)
    assert reader.header.size == 10
    assert np.all(np.equal(reader.header["tilt_alpha"], angle))
    assert np.all(np.equal(reader.header.position, position))


def test_write_images(tmp_path, io_test_data):
    filename = os.path.join(tmp_path, "tmp_%03d.png")

    data, angle, position = io_test_data

    def test(vmin, vmax):
        writer = parakeet.io.new(filename, shape=data.shape, vmin=vmin, vmax=vmax)
        for i in range(data.shape[0]):
            writer.data[i, :, :] = data[i, :, :]
            writer.header[i]["tilt_alpha"] = angle[i]
            writer.header[i]["shift_x"] = position[i][0]
            writer.header[i]["shift_y"] = position[i][1]
            writer.header[i]["stage_z"] = position[i][2]

        assert writer.shape == data.shape
        assert writer.is_mrcfile_writer == False
        assert writer.is_nexus_writer == False
        assert writer.is_image_writer == True
        assert writer.vmin == vmin
        assert writer.vmax == vmax
        if vmin is not None and vmax is not None:
            writer.vmin = vmin + 1
            writer.vmax = vmax - 1
            assert writer.vmin == vmin + 1
            assert writer.vmax == vmax - 1

        for i in range(data.shape[0]):
            assert os.path.exists(filename % (i + 1))

    test(None, None)
    test(np.min(data), np.max(data))


def test_unknown_image(tmp_path):
    filename = os.path.join(tmp_path, "tmp.unknown")

    with pytest.raises(RuntimeError):
        writer = parakeet.io.new(filename, shape=(1, 10, 10))

    with pytest.raises(RuntimeError):
        reader = parakeet.io.open(filename)
