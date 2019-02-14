import nose.tools
import numpy as np
import scipy.ndimage as ndimage

import skeletonization.skeleton.vessel_phantom as vessel_phantom


def test_volume_bool_to_dtype():
    bstack = np.ones((4, 4, 4), dtype=bool)
    bstack[:2, :2, :] = False
    out = vessel_phantom.volume_bool_to_dtype(bstack)
    nose.tools.assert_equal(out.dtype, np.uint8)
    np.testing.assert_array_equal(np.unique(out), np.array([0, 255]))

    out = vessel_phantom.volume_bool_to_dtype(bstack, fg=50)
    np.testing.assert_array_equal(np.unique(out), np.array([0, 50]))

    out = vessel_phantom.volume_bool_to_dtype(bstack, dtype=np.int16)
    nose.tools.assert_equal(out.dtype, np.int16)


def test_cylinder_on_axis():
    # test x axis
    cylinder_radius = 5
    mask = vessel_phantom.cylinder_on_axis(radius=cylinder_radius, shape=(32, 32, 32))
    # get 1D slice through cylinder, count ON pixels
    output_radius = (mask[16, 16, :] != 0).sum() / 2
    np.testing.assert_array_equal(output_radius, cylinder_radius)

    # test y axis
    cylinder_radius = 11
    mask = vessel_phantom.cylinder_on_axis(radius=cylinder_radius, axis=1, shape=(32, 32, 32))
    # get 1D slice through cylinder, count ON pixels
    output_radius = (mask[:, 16, 16] != 0).sum() / 2
    np.testing.assert_array_equal(output_radius, cylinder_radius)

    # test z axis
    cylinder_radius = 1
    mask = vessel_phantom.cylinder_on_axis(radius=cylinder_radius, axis=2, shape=(32, 32, 32))
    # get 1D slice through cylinder, count ON pixels
    output_radius = (mask[16, :, 16] != 0).sum() / 2
    np.testing.assert_array_equal(output_radius, cylinder_radius)

    with nose.tools.assert_raises(AssertionError):
        vessel_phantom.cylinder_on_axis(axis=3)


def test_add_cylinder_px():
    shape = (32, 32, 32)
    volume = np.zeros(shape, dtype=np.uint8)
    point1 = (0, 0, 0)
    point2 = (31, 31, 31)
    radius = 5
    cyl = vessel_phantom.add_cylinder_px(volume, point1, point2, radius)

    nose.tools.assert_equal(cyl.dtype, bool)
    np.testing.assert_array_equal(cyl, cyl[::-1, ::-1, ::-1])

    for i in range(shape[0]):
        nose.tools.assert_equal(
            cyl[i, i, i], True,
            msg="voxel [{}, {}, {}]".format(i, i, i))

    point1 = (15, 15, 15)
    point2 = (0, 31, 31)
    radius = 2
    cyl2 = vessel_phantom.add_cylinder_px(cyl, point1, point2, radius)
    # is our second cylinder getting to it's target coord?
    nose.tools.assert_equal(cyl2[point2], True)
    # is our first cylinder still there?
    for i in range(shape[0]):
        nose.tools.assert_equal(
            cyl[i, i, i], True,
            msg="voxel [{}, {}, {}]".format(i, i, i))


def test_add_cylinder_basecoord():
    shape = (32, 32, 32)
    volume = np.zeros(shape, dtype=np.uint8)
    cylinder = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), 0.25)
    point1, point2, radius = cylinder
    cyl = vessel_phantom.add_cylinder_basecoord(volume, point1, point2, radius)
    nose.tools.assert_equal(cyl.dtype, bool)
    np.testing.assert_array_equal(cyl, cyl[::-1, ::-1, ::-1])


def test_cylinders_px_to_basecoord():
    shape = (32, 32, 32)
    cylinder_list = [
        ((0, 0, 0), (31, 31, 31), 4),
        ((31, 15, 31), (16, 16, 16), 10)
    ]
    cylinder_list_float = vessel_phantom.cylinders_px_to_basecoord(cylinder_list, shape)
    nose.tools.assert_equal(len(cylinder_list_float), len(cylinder_list))
    nose.tools.assert_equal(len(cylinder_list_float[0]), len(cylinder_list[0]))
    expected_first_cylinder = ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), 0.25)
    nose.tools.assert_equal(cylinder_list_float[0], expected_first_cylinder)


def test_cylinders_base_to_pixelcoord():
    shape = (32, 32, 32)
    cylinder_list = [
        ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), 0.25),
        ((0.096, 0.080, 0.840), (0.096, 0.080, 0.840), 0.066),
    ]
    cylinder_list_px = vessel_phantom.cylinders_base_to_pixelcoord(cylinder_list, shape)
    nose.tools.assert_equal(len(cylinder_list_px), len(cylinder_list))
    nose.tools.assert_equal(len(cylinder_list_px[0]), len(cylinder_list[0]))
    expected_first_cylinder = ((0, 0, 0), (31, 31, 31), 4)
    nose.tools.assert_equal(cylinder_list_px[0], expected_first_cylinder)


def test_create_cylinders_volume():
    shape = (32, 32, 32)
    # these two cylinders were manually checked to NOT touch each other
    cylinder_list = [
        ((0, 0, 0), (31, 31, 31), 4),
        ((10, 0, 25), (10, 31, 25), 4)
    ]
    volume = vessel_phantom.create_cylinders_volume(shape, cylinder_list)
    foreground = 1

    # we are checking correctness of cylinder elsewhere,
    # so we focus on that the listed cylinder is generally in the right place
    for p1, p2, r in cylinder_list:
        nose.tools.assert_equal(volume[p1], foreground)
        nose.tools.assert_equal(volume[p2], foreground)

    # check that there are only as many cylinders as what we draw
    connectivity_kernel = ndimage.generate_binary_structure(volume.ndim, 2)
    _, countObjects = ndimage.measurements.label(volume, connectivity_kernel)
    np.testing.assert_array_equal(countObjects, len(cylinder_list))

    # confirm default outputs aren't unexpected
    nose.tools.assert_equals(volume.dtype, np.uint8)
    np.testing.assert_array_equal(np.unique(volume), np.array([0, foreground]))


def test_create_cylinders_volume_foreground():
    shape = (32, 32, 32)
    cylinder_list = [
        ((0, 0, 0), (31, 31, 31), 4),
        ((10, 0, 25), (10, 31, 25), 4)
    ]
    foreground = 16
    volume = vessel_phantom.create_cylinders_volume(shape, cylinder_list, foreground=foreground)
    np.testing.assert_array_equal(np.unique(volume), np.array([0, foreground]))


def test_create_cylinders_volume_dtype():
    shape = (32, 32, 32)
    cylinder_list = [
        ((0, 0, 0), (31, 31, 31), 4),
        ((10, 0, 25), (10, 31, 25), 4)
    ]
    dtype = np.int16
    volume = vessel_phantom.create_cylinders_volume(shape, cylinder_list, dtype=dtype)
    nose.tools.assert_equals(volume.dtype, dtype)

    dtype = bool
    volume = vessel_phantom.create_cylinders_volume(shape, cylinder_list, dtype=dtype)
    nose.tools.assert_equals(volume.dtype, dtype)


def test_z_cylinder():
    shape = (64, 64, 64)
    radius = shape[0] // 4
    volume, cylinder = vessel_phantom.z_cylinder(shape, radius)
    nose.tools.assert_equals(volume.dtype, np.uint8)
    np.testing.assert_array_equal(np.unique(volume), np.array([0, 1]))

    connectivity_kernel = ndimage.generate_binary_structure(volume.ndim, 2)
    _, countObjects = ndimage.measurements.label(volume, connectivity_kernel)
    nose.tools.assert_equals(countObjects, 1)  # there are 3 cylinders in this volume
    nose.tools.assert_equals(len(cylinder), 1)  # we should have 3 vectoriuzed cylinders as well
    nose.tools.assert_equals(cylinder[0][0], (shape[0] // 2, shape[0] // 2, 0))
    nose.tools.assert_equals(cylinder[0][1], (shape[0] // 2, shape[0] // 2, shape[2] - 1))
    nose.tools.assert_equals(cylinder[0][2], shape[0] // 4)


def test_z_cylinder_asymmetric():
    shape = (10, 20, 5)
    radius = shape[0] // 4
    volume, cylinder = vessel_phantom.z_cylinder(shape, radius)
    nose.tools.assert_equals(len(cylinder), 1)  # we should have 3 vectoriuzed cylinders as well
    nose.tools.assert_equals(cylinder[0][0], (shape[0] // 2, shape[1] // 2, 0))
    nose.tools.assert_equals(cylinder[0][1], (shape[0] // 2, shape[1] // 2, shape[2] - 1))
    nose.tools.assert_equals(cylinder[0][2], shape[0] // 4)


def test_vertical_cylinders():
    tile_size = 64
    shape = (3 * tile_size, 3 * tile_size, 10)
    mask, cylinders = vessel_phantom.vertical_cylinders(shape[0], shape[2])

    _, countObjects = ndimage.measurements.label(
        mask, ndimage.generate_binary_structure(mask.ndim, 2))
    nose.tools.assert_equals(countObjects, 3)  # there are 3 cylinders in this volume
    nose.tools.assert_equals(len(cylinders), 3)  # we should have 3 vectoriuzed cylinders as well
    nose.tools.assert_equals(cylinders[0][0], (shape[0] // 2, shape[0] // 2, 0))
    nose.tools.assert_equals(cylinders[0][1], (shape[0] // 2, shape[0] // 2, shape[2] - 1))
    nose.tools.assert_equals(cylinders[0][2], tile_size // 4)


def test_vessel_diagonal():
    cube_edge = 50
    mask = vessel_phantom.diagonal(cube_edge=cube_edge, radius=5)
    np.testing.assert_array_equal(tuple(np.unique(mask)), (0, 1))
    count_object_voxels = 5804  # the expected number of voxels on in this object
    np.testing.assert_array_equal(mask.sum(), count_object_voxels)
    connectivity_kernel = ndimage.generate_binary_structure(mask.ndim, 2)
    _, countObjects = ndimage.measurements.label(mask, connectivity_kernel)
    np.testing.assert_array_equal(countObjects, 1)  # there should only be one object

    for i in range(cube_edge):
        nose.tools.assert_equal(
            mask[i, i, i], True,
            msg="voxel [{}, {}, {}]".format(i, i, i))

    with nose.tools.assert_raises(ValueError):
        mask = vessel_phantom.diagonal(cube_edge=10, radius=11)


def test_vessel_loop():
    cube_edge = 50
    mask = vessel_phantom.vessel_loop(cube_edge=cube_edge)
    np.testing.assert_array_equal(mask.shape, (cube_edge, cube_edge, cube_edge))
    np.testing.assert_array_equal(tuple(np.unique(mask)), (0, 1))
    count_object_voxels = 3809  # the expected number of voxels on in this object
    np.testing.assert_array_equal(mask.sum(), count_object_voxels)
    _, countObjects = ndimage.measurements.label(
        mask, ndimage.generate_binary_structure(mask.ndim, 2))
    np.testing.assert_array_equal(countObjects, 1)  # there is 1 separate object in this volume


def test_vessel_tree():
    cube_edge = 50
    mask = vessel_phantom.vessel_tree(cube_edge=cube_edge)
    np.testing.assert_array_equal(mask.shape, (cube_edge, cube_edge, cube_edge))
    np.testing.assert_array_equal(tuple(np.unique(mask)), (0, 1))
    count_object_voxels = 4776  # the expected number of voxels on in this object
    np.testing.assert_array_equal(mask.sum(), count_object_voxels)
