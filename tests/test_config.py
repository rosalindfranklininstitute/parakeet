import os
import pytest
import parakeet.config


def dict_approx_equal(a, b):
    def walk(a, b):
        if isinstance(a, dict):
            if a.keys() != b.keys():
                return False
            return all([walk(a[k], b[k]) for k in a.keys()])
        elif isinstance(a, list):
            if len(a) != len(b):
                return False
            return all([walk(a[i], b[i]) for i in range(len(a))])
        else:
            print(a, b, a == pytest.approx(b))
            return a == pytest.approx(b)

    return walk(a, b)


def test_temp_directory():
    assert parakeet.config.temp_directory() == "_parakeet"


def test_default():
    config = parakeet.config.default()


def test_save(tmp_path):
    filename = os.path.join(tmp_path, "tmp-save.yaml")
    config = parakeet.config.default()
    parakeet.config.save(config, filename)


def test_new(tmp_path):
    filename = os.path.join(tmp_path, "tmp.yaml")
    config = parakeet.config.new(filename)


def test_edit(tmp_path):
    filename = os.path.join(tmp_path, "tmp.yaml")
    config = parakeet.config.new(filename)
    config = parakeet.config.edit(
        filename,
        config_obj="""
        sample:
            box:
                - 2000
                - 2000
                - 2000
    """,
    )
    config = parakeet.config.edit(
        filename,
        config_obj="""
        sample:
            molecules:
                pdb:
                    - id: 4v1w
                      instances: 1
    """,
    )
    assert config.sample.molecules.pdb[0].id == "4v1w"
    assert config.sample.box[0] == 2000


def test_load(tmp_path):
    filename = os.path.join(tmp_path, "tmp.yaml")
    config = parakeet.config.new(filename)
    config = parakeet.config.load(filename)

    assert dict_approx_equal(config.dict(), parakeet.config.default().dict())


def test_show():
    parakeet.config.show(parakeet.config.Config())
