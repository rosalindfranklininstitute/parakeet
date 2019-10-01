import numpy
import os
import pytest
import elfantasma.io


@pytest.fixture
def io_test_data():

    # The shape of the data
    shape = (10, 100, 100)

    # Generate the data
    data = numpy.random.randint(low=0, high=100, size=shape[0] * shape[1] * shape[2])
    data.shape = shape

    # Generate the angles
    angle = numpy.array([x for x in range(0, shape[0])])

    # Generate the positons
    position = numpy.array([[x, x, x] for x in range(0, shape[0])])

    # Return the data
    return (data, angle, position)


def test_read_write_mrcfile(tmp_path, io_test_data):

    filename = os.path.join(tmp_path, "tmp.mrc")

    data, angle, position = io_test_data

    writer = elfantasma.io.new(filename, shape=data.shape)
    for i in range(data.shape[0]):
        writer.data[i, :, :] = data[i, :, :]
        writer.angle[i] = angle[i]
        writer.position[i] = position[i]

    assert writer.shape == data.shape

    # Test different ways of writing position
    writer.position[0, 0] = position[0][0]
    writer.position[0, 1] = position[0][1]
    writer.position[0, 2] = position[0][2]
    writer.position[2, :] = position[2]
    writer.position[3, 0:2] = position[3][0:2]
    writer.position[3, 2] = position[3][2]

    # Make sure stuff is written
    writer = None

    reader = elfantasma.io.open(filename)
    assert reader.data.shape == (10, 100, 100)
    assert reader.angle.shape == (10,)
    assert numpy.all(numpy.equal(reader.angle, angle))
    assert numpy.all(numpy.equal(reader.position, position))


def test_write_nexus(tmp_path, io_test_data):

    filename = os.path.join(tmp_path, "tmp.h5")

    data, angle, position = io_test_data

    writer = elfantasma.io.new(filename, shape=data.shape)
    for i in range(data.shape[0]):
        writer.data[i, :, :] = data[i, :, :]
        writer.angle[i] = angle[i]
        writer.position[i] = position[i]

    assert writer.shape == data.shape

    # Test different ways of writing position
    writer.position[0, 0] = position[0][0]
    writer.position[0, 1] = position[0][1]
    writer.position[0, 2] = position[0][2]
    writer.position[2, :] = position[2]
    writer.position[3, 0:2] = position[3][0:2]
    writer.position[3, 2] = position[3][2]

    # Make sure stuff is written
    writer = None

    reader = elfantasma.io.open(filename)
    assert reader.data.shape == (10, 100, 100)
    assert reader.angle.shape == (10,)
    assert numpy.all(numpy.equal(reader.angle, angle))
    assert numpy.all(numpy.equal(reader.position, position))


def test_write_images(tmp_path, io_test_data):

    filename = os.path.join(tmp_path, "tmp_%03d.png")

    data, angle, position = io_test_data

    def test(vmin, vmax):
        writer = elfantasma.io.new(filename, shape=data.shape, vmin=vmin, vmax=vmax)
        for i in range(data.shape[0]):
            writer.data[i, :, :] = data[i, :, :]
            writer.angle[i] = angle[i]
            writer.position[i] = position[i]

        assert writer.shape == data.shape

        for i in range(data.shape[0]):
            assert os.path.exists(filename % (i + 1))

    test(None, None)
    test(numpy.min(data), numpy.max(data))
