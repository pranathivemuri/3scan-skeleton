import os

import nose.tools
import numpy as np

import skeleton.io_tools as io_tools
from tempfile import TemporaryDirectory


def test_module_dir():
    d = io_tools.module_dir()
    assert d.endswith('skeleton'), d


def test_module_relative_path():
    nose.tools.assert_equals(
        io_tools.module_relative_path('io_tools_tests.py'),
        __file__)


def test_pad_int():
    np.testing.assert_array_equal(io_tools.padInt(5), "00000005")


def test_force_directory():
    with TemporaryDirectory() as td:
        # Already exists as dir, ok
        io_tools.force_directory(td)

        # Creates. ok
        io_tools.force_directory(os.path.join(td, "foo"))
        io_tools.force_directory(os.path.join(td, "foo/bar/baz"))

        # Already exists, not dir, asserts
        baz_file = os.path.join(td, "baz")
        with open(baz_file, "w") as f:
            f.write("placeholder")

        with nose.tools.assert_raises(AssertionError):
            io_tools.force_directory(baz_file)
