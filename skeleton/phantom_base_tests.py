import nose.tools
import numpy as np

import skeleton.phantom_base as phantom_base


def test_normalized_L2():
    A = np.array([[-1, -1, -1],
                  [0, 2, 0]])
    norm = phantom_base.normalized(A, order=2)
    L2_norm = (A.T / np.sqrt((A ** 2).sum(axis=1))).T
    np.testing.assert_array_equal(norm, L2_norm)

    norm = phantom_base.normalized(A, axis=0, order=2)
    L2_norm = A / np.sqrt(((A ** 2).sum(axis=0)))
    np.testing.assert_array_equal(norm, L2_norm)

    # single vector
    A = np.array([2, 1, 0])
    norm = phantom_base.normalized(A, order=2)
    np.testing.assert_array_equal(norm, A / np.sqrt((A ** 2).sum()))


def test_normalized_L1():
    A = np.array([[-1, -1, -1],
                  [0, 2, 0]])
    norm = phantom_base.normalized(A, order=1)
    L2_norm = (A.T / np.abs(A).sum(axis=1)).T
    np.testing.assert_array_equal(norm, L2_norm)

    norm = phantom_base.normalized(A, axis=0, order=1)
    L2_norm = A / np.abs(A).sum(axis=0)
    np.testing.assert_array_equal(norm, L2_norm)

    # single vector
    A = np.array([2, 1, 0])
    norm = phantom_base.normalized(A, order=1)
    np.testing.assert_array_equal(norm, A / np.abs(A.sum()))


def test_normalized_lengthzero():
    A = np.array([0, 0, 0])
    norm = phantom_base.normalized(A)
    np.testing.assert_array_equal(norm, A)


def test_to_basecoord():
    shape = 32
    p = phantom_base.to_basecoord(0, shape)
    nose.tools.assert_equal(p, -1.0)

    p = phantom_base.to_basecoord(31, shape)
    nose.tools.assert_equal(p, 1.0)

    x = 15
    p = phantom_base.to_basecoord(x, shape)
    ds = 2 / (shape - 1)
    expected = x * ds - 1
    nose.tools.assert_equal(p, expected)

    with nose.tools.assert_raises(AssertionError):
        p = phantom_base.to_basecoord(32, 32)

    with nose.tools.assert_raises(AssertionError):
        p = phantom_base.to_basecoord(-1, 32)

    # test single dimension
    p = phantom_base.to_basecoord(0, 1)
    nose.tools.assert_equal(p, -1.0)


def test_scale_point_to_basecoord():
    shape = (32, 32, 32)
    ds = 2 / (np.array(shape) - 1)  # sample step in base coordinates

    point = (0, 0, 0)
    scaled = phantom_base.scale_point_to_basecoord(point, shape)
    nose.tools.assert_equal(scaled, (-1.0, -1.0, -1.0))

    point = (31, 31, 31)
    scaled = phantom_base.scale_point_to_basecoord(point, shape)
    nose.tools.assert_equal(scaled, (1.0, 1.0, 1.0))

    point = (15, 15, 15)
    scaled = phantom_base.scale_point_to_basecoord(point, shape)
    expected = tuple(p * dp - 1 for p, dp in zip(point, ds))
    np.testing.assert_almost_equal(scaled, expected, decimal=6)

    point = (18, 0, 8)
    scaled = phantom_base.scale_point_to_basecoord(point, shape)
    expected = tuple(p * dp - 1 for p, dp in zip(point, ds))
    np.testing.assert_almost_equal(scaled, expected, decimal=6)

    point = (1, 1, 1)
    scaled = phantom_base.scale_point_to_basecoord(point, shape)
    expected = tuple(p * dp - 1 for p, dp in zip(point, ds))
    np.testing.assert_almost_equal(scaled, expected, decimal=6)


def test_scale_radius_to_basecoord():
    shape = (32, 32, 32)
    r = phantom_base.scale_radius_to_basecoord(5, shape)
    target = 0.3125
    nose.tools.assert_equals(r, target)

    shape = (32, 32, 1)
    r = phantom_base.scale_radius_to_basecoord(5, shape)
    nose.tools.assert_equals(r, target)

    with nose.tools.assert_raises(AssertionError):
        r = phantom_base.scale_radius_to_basecoord(-1, shape)


def test_to_pixelcoord():
    shape = 32
    p = phantom_base.to_pixelcoord(-1.0, shape)
    nose.tools.assert_equal(p, 0)

    p = phantom_base.to_pixelcoord(1.0, shape)
    nose.tools.assert_equal(p, 31)

    x = 0.0
    p = phantom_base.to_pixelcoord(x, shape)
    expected = 16
    nose.tools.assert_equal(p, expected)

    # test single dimension
    p = phantom_base.to_pixelcoord(-1.0, 1)
    nose.tools.assert_equal(p, 0)


def test_scale_point_to_pixelcoord():
    shape = (32, 32, 32)
    ds = 2 / (np.array(shape) - 1)  # sample step in base coordinates

    point = (-1.0, -1.0, -1.0)
    scaled = phantom_base.scale_point_to_pixelcoord(point, shape)
    nose.tools.assert_equal(scaled, (0, 0, 0))

    point = (1.0, 1.0, 1.0)
    scaled = phantom_base.scale_point_to_pixelcoord(point, shape)
    nose.tools.assert_equal(scaled, (31, 31, 31))

    point = (0.0, 0.0, 0.0)
    expected = (16, 16, 16)
    scaled = phantom_base.scale_point_to_pixelcoord(point, shape)
    np.testing.assert_almost_equal(scaled, expected, decimal=6)

    expected = (18, 0, 8)
    point = tuple(p * dp - 1 for p, dp in zip(expected, ds))
    scaled = phantom_base.scale_point_to_pixelcoord(point, shape)
    np.testing.assert_almost_equal(scaled, expected, decimal=6)

    expected = (1, 1, 1)
    point = tuple(p * dp - 1 for p, dp in zip(expected, ds))
    scaled = phantom_base.scale_point_to_pixelcoord(point, shape)
    np.testing.assert_almost_equal(scaled, expected, decimal=6)


def test_scale_radius_to_pixelcoord():
    shape = (32, 32, 32)
    r = phantom_base.scale_radius_to_pixelcoord(0.25, shape)
    target = 4
    nose.tools.assert_equals(r, target)

    shape = (32, 64, 1)
    r = phantom_base.scale_radius_to_pixelcoord(0.25, shape)
    nose.tools.assert_equals(r, target)

    with nose.tools.assert_raises(AssertionError):
        r = phantom_base.scale_radius_to_pixelcoord(-1, shape)


def assert_coord_block_properties(x, y, z):
    for coord in [x, y, z]:
        nose.tools.assert_equal(coord.min(), -1)
        nose.tools.assert_equal(coord.max(), 1)

    x_slice = x[:, 0, 0].flatten()
    y_slice = y[0, :, 0].flatten()
    z_slice = z[0, 0, :].flatten()

    np.testing.assert_array_equal(x_slice, y_slice)
    np.testing.assert_array_equal(y_slice, z_slice)


def test_create_coordinate_block():
    x, y, z = phantom_base.create_coordinate_block(shape=(5, 5, 5))
    assert_coord_block_properties(x, y, z)

    x, y, z = phantom_base.create_coordinate_block(shape=(10, 10, 10))
    assert_coord_block_properties(x, y, z)

    shape = (3, 5, 10)
    x, y, z = phantom_base.create_coordinate_block(shape=shape)

    nose.tools.assert_tuple_equal(x.shape, shape)
    nose.tools.assert_tuple_equal(y.shape, shape)
    nose.tools.assert_tuple_equal(z.shape, shape)


def test_trace_function_minimal():
    call_step_list = []

    def dumb_func(t):
        call_step_list.append(t)
        return 0, 0, 0, 0

    t_shape = (20, 20, 20)
    traced = phantom_base.trace_function(dumb_func, shape=t_shape, step_count=5)

    # Check how the function was called
    nose.tools.assert_equal(min(call_step_list), 0)
    nose.tools.assert_equal(max(call_step_list), 1)

    # Check that this function returns empty (no trace)
    np.testing.assert_array_equal(traced, np.zeros(t_shape, dtype=bool))


def test_trace_function_point():
    def point_func(t):
        return 0, 0, 0, 1

    # This should result in a ball with radius equal to the box size
    t_shape = (64, 64, 64)
    traced = phantom_base.trace_function(point_func, shape=t_shape, step_count=1)
    traced[traced == 255] = 1

    unit_sphere = (4 / 3) * np.pi
    domain_size = 2 ** 3  # domain is volume of [-1:1, -1:1, -1:1], or 8 unit cubes
    expected_fill_percent = unit_sphere / domain_size

    # Error producing the unit-sphere is < 5%
    err = abs(traced.mean() - expected_fill_percent)
    nose.tools.assert_less(err, 0.05)


def test_trace_function_isotropy():
    def point_func(t):
        return 0, 0, 0, 1

    t_shape = (64, 64, 86)
    iso = (1, 1, t_shape[0] / t_shape[2])
    traced = phantom_base.trace_function(point_func, shape=t_shape, isotropy=iso, step_count=1)

    # sphere diameters should be nearly equal in all axes
    diameter_x = np.diff(
        phantom_base.contiguous_elements(traced[:, t_shape[1] // 2, t_shape[2] // 2])).flatten()[0]
    diameter_y = np.diff(
        phantom_base.contiguous_elements(traced[t_shape[0] // 2, :, t_shape[2] // 2])).flatten()[0]
    diameter_z = np.diff(
        phantom_base.contiguous_elements(traced[t_shape[0] // 2, t_shape[1] // 2, :])).flatten()[0]
    sphere_diameters = [diameter_x, diameter_y, diameter_z]
    # minus 2 on the XY because of incomplete filling on edge
    np.testing.assert_array_equal(sphere_diameters, [t_shape[0] - 2, t_shape[1] - 2, t_shape[0]])


def test_cylinder_in_x():
    shape = (20, 20, 20)
    cyl = phantom_base.cylinder_in_x(1, shape=shape, step_count=40)
    np.testing.assert_array_equal(cyl, cyl[::-1, :, :])

    nose.tools.assert_true(np.all(cyl[:, 10, 10] == 255))
    nose.tools.assert_true(np.all(cyl[:, 0, 0] == 0))
    nose.tools.assert_true(np.all(cyl[:, 19, 19] == 0))

    # All steps in x should be identical
    for step in range(shape[2]):
        np.testing.assert_array_equal(
            cyl[0, :, :],
            cyl[step, :, :]
        )

    mt = phantom_base.cylinder_in_x(0, shape=(20, 20, 20), step_count=1)
    np.testing.assert_array_equal(mt, 0)


def test_cylinder_in_xy():
    cyl = phantom_base.cylinder_in_xy(1, shape=(20, 20, 20), step_count=40)
    np.testing.assert_array_equal(cyl, cyl[::-1, ::-1, :])

    np.testing.assert_array_equal(cyl[:, :, 0], 0)
    np.testing.assert_array_equal(cyl[:, :, -1], 0)

    mt = phantom_base.cylinder_in_xy(0, shape=(20, 20, 20), step_count=1)
    np.testing.assert_array_equal(mt, 0)


def test_cylinder_in_xyz():
    cyl = phantom_base.cylinder_in_xyz(1, shape=(20, 20, 20), step_count=40)
    np.testing.assert_array_equal(cyl, cyl[::-1, ::-1, ::-1])

    for i in range(20):
        nose.tools.assert_equal(cyl[i, i, i], 255)

    mt = phantom_base.cylinder_in_xyz(0, shape=(20, 20, 20), step_count=1)
    np.testing.assert_array_equal(mt, 0)


def test_torus_xy():
    torus = phantom_base.torus_xy(0.1, 0.5, shape=(50, 50, 50))

    np.testing.assert_array_equal(
        torus,
        torus[::-1, :, :]
    )

    np.testing.assert_array_equal(
        torus,
        torus[:, ::-1, :]
    )


def _find_diameter(arr):
    """
    Return the diameter across a 1D slice
    This slice should go through the cross section centroid
    The diameter is the distance between the [first_nonzero, last_nonzero)
    in a 1D array. Note that the last coordinate is exclusive, which
    makes the distance calculation correct

    Input:
        arr - ndarray 1d, containing zeros and nonzero values

    See kesm.math.statistics.nonzero_bounds for more info
    """
    bounds = phantom_base.nonzero_bounds(arr)
    return bounds[1] - bounds[0]


def test_contiguous_elements():
    arr = np.array([0, 1, 3, 3, 3, 2, 2])
    np.testing.assert_array_equal(
        phantom_base.contiguous_elements(arr == 3),
        np.array([[2, 5]]))
    np.testing.assert_array_equal(
        phantom_base.contiguous_elements(arr == 2),
        np.array([[5, 7]]))


def test_cylinder_xaxis():
    # test single axis cylinder, parallel to the x axis
    shape = (64, 64, 64)
    halfshape = tuple(s // 2 for s in shape)
    p1 = (1, 0, 0)
    p2 = (-1, 0, 0)
    r = 0.5  # in base coordinates; radius will be 1/4 the shape
    volume = phantom_base.cylinder(shape, p1, p2, r)

    # check the centroid and radius
    xslice = volume[halfshape[0], :, :]
    row = xslice[halfshape[1], :]
    col = xslice[:, halfshape[2]]
    # given a radius in real numbers, calculate that expected radius in pixels
    expected_diameters = tuple(int(s * r) for s in shape)
    # assert that a row slice through the object centroid has the correct radius
    nose.tools.assert_equals(_find_diameter(row), expected_diameters[0])
    # assert that a column slice through the object centroid has the correct radius
    nose.tools.assert_equals(_find_diameter(col), expected_diameters[1])
    # make sure that the center line of the cylinder exists across the volume
    expected_line = np.ones(shape[0], dtype=bool)
    np.testing.assert_array_equal(
        volume[:, halfshape[1], halfshape[2]],
        expected_line)

    # check that cap == "none" is identical with full axis
    p1 = (-0.5, 0, 0)
    p2 = (0.5, 0, 0)
    r = 0.5  # in base coordinates; radius will be 1/4 the shape
    vol_nocap = phantom_base.cylinder(shape, p1, p2, r, cap="none")
    np.testing.assert_array_equal(vol_nocap, volume)


def test_cylinder_xaxis_flatcap():
    # cylinder along the xaxis, but with flat caps
    shape = (64, 64, 64)
    halfshape = tuple(s // 2 for s in shape)
    p1 = (-0.5, 0, 0)
    p2 = (0.5, 0, 0)
    r = 0.5  # in base coordinates; radius will be 1/4 the shape
    volume = phantom_base.cylinder(shape, p1, p2, r, cap="flat")
    # get the expected centerline of the cylinder, with a planar cap
    # Parts of the line outside of the planes defined by the two points
    # through the center line should not be turned on.
    expected_line = np.zeros(shape[0], dtype=bool)
    expected_line[(halfshape[0] // 2):(halfshape[0] + halfshape[0] // 2)] = 1
    np.testing.assert_array_equal(volume[:, 31, 31], expected_line)


def test_cylinder_xaxis_spherecap():
    # cylinder along the xaxis, but with flat caps
    shape = (64, 64, 64)
    halfshape = tuple(s // 2 for s in shape)
    p1 = (-0.5, 0, 0)
    p2 = (0.5, 0, 0)
    r = 0.25  # in base coordinates; radius will be 1/4 the shape
    # radius in pixels; divide by 2 since space is [-1, 1]
    rpix = int(shape[0] * r / 2)
    volume = phantom_base.cylinder(shape, p1, p2, r, cap="sphere")
    center_line = volume[:, halfshape[1], halfshape[2]]
    # create the expected centerline between point1 and point2, plus the
    # spherical radius of each of the endcaps at both endpoints
    expected_line = np.zeros(shape[0], dtype=bool)
    expected_line[(halfshape[0] // 2 - rpix):(halfshape[0] + halfshape[0] // 2 + rpix)] = 1
    np.testing.assert_array_equal(center_line, expected_line)


def test_cylinder_tiny_radius():
    # check that we are logging a warning for radii smaller than grid resolution
    shape = (10, 10, 10)
    p1 = (-0.5, 0, 0)
    p2 = (0.5, 0, 0)
    r = 0.05  # use an epsilon-sized radius, should be smaller than the grid size
    volume = phantom_base.cylinder(shape, p1, p2, r)
    # volume should be all zeros
    np.testing.assert_array_equal(np.unique(volume), np.zeros(1))

    r = 1 / min(shape)  # use the grid size, see what happens
    volume = phantom_base.cylinder(shape, p1, p2, r)
    # volume should be all zeros
    np.testing.assert_array_equal(np.unique(volume), np.zeros(1))


def test_cylinder_big_radius():
    shape = (10, 10, 10)
    p1 = (0, 0, -1)
    p2 = (0, 0, 1)
    r = 2
    volume = phantom_base.cylinder(shape, p1, p2, r)
    np.testing.assert_array_equal(np.unique(volume), np.ones(1))


def test_cylinder_offset_from_axis():
    # test cylinder line in negative quadrant
    shape = (64, 64, 64)
    quartershape = tuple(s // 4 for s in shape)
    p1 = (-.9, -0.5, -0.5)
    p2 = (-0.2, -0.5, -0.5)
    r = 0.2
    volume = phantom_base.cylinder(shape, p1, p2, r, cap="flat")
    expected_line = np.zeros(shape[0], dtype=bool)
    expected_line[4:26] = 1
    np.testing.assert_array_equal(
        volume[:, quartershape[1], quartershape[2]],
        expected_line)

    xindex = quartershape[0]
    xslice = volume[xindex, :, :]
    row = xslice[quartershape[1], :]
    col = xslice[:, quartershape[2]]
    expected_diameters = tuple(int(s * r) for s in shape)
    nose.tools.assert_almost_equal(_find_diameter(row), expected_diameters[0], delta=1)
    nose.tools.assert_almost_equal(_find_diameter(col), expected_diameters[1], delta=1)


def test_cylinder_xy_inf():
    # make sure that we can generate an arbitrary cylinder across 2 axes
    shape = (64, 64, 64)
    p1 = (-1, -1, 0)
    p2 = (1, 1, 0)
    r = 0.5  # in base coordinates; radius will be 1/4 the shape
    volume = phantom_base.cylinder(shape, p1, p2, r, cap="none")
    np.testing.assert_array_equal(volume, volume[::-1, ::-1, :])


def test_cylinder_xyz_inf():
    # make sure that we can generate an arbitrary cylinder across 3 dimensions
    shape = (64, 64, 64)
    p1 = (-1, -1, -1)
    p2 = (1, 1, 1)
    r = 0.5  # in base coordinates; radius will be 1/4 the shape
    volume = phantom_base.cylinder(shape, p1, p2, r, cap="none")
    np.testing.assert_array_equal(volume, volume[::-1, ::-1, ::-1])


def test_cylinder_xyz_flatcap():
    # test the planar cut on generating a cylinder across 3 dimensions
    shape = (64, 64, 64)
    quarter = np.array([s // 4 for s in shape])
    p1 = (-0.5, -0.5, -0.5)
    p2 = (0.5, 0.5, 0.5)
    r = 0.25  # in base coordinates; radius will be 1/4 the shape
    volume = phantom_base.cylinder(shape, p1, p2, r)
    np.testing.assert_array_equal(volume, volume[::-1, ::-1, ::-1])

    # make sure we have flattened at the edge of the cylinder
    # get the diagonal line through the cylinder, then check bounds
    line = np.array([volume[x, x, x] for x in range(shape[0])])
    nose.tools.assert_equal(phantom_base.nonzero_bounds(line),
                            (quarter[0], shape[0] - quarter[0]))


def test_cylinder_cap_strmatch_assertion():
    # invalid cap types should fail
    shape = (4, 4, 4)
    cyl = ((0, 0, 0), (1, 1, 1), .5)
    with nose.tools.assert_raises(AssertionError):
        phantom_base.cylinder(shape, *cyl, cap="foo")


def test_distance_point_from_plane_sign_assertion():
    plane = np.array([1, 1, 0])
    point = np.array([0, -1, 0])
    points = np.array([0, -2, 0])

    with nose.tools.assert_raises(AssertionError):
        phantom_base.distance_point_from_plane(points, plane, point, 0)

    with nose.tools.assert_raises(AssertionError):
        phantom_base.distance_point_from_plane(points, plane, point, 2)

    with nose.tools.assert_raises(AssertionError):
        phantom_base.distance_point_from_plane(points, plane, point, -3.5)


def test_distance_point_from_plane():
    # test core method that returns correct distance of point from plane
    points = np.array([
        [0, -1, 0],  # will be new origin
        [0, -1, 1],  # on the plane
        [0, -1, -5],  # still on the plane
        [0, 0, 0],  # should be on the positive side of plane
        [2, 0, 0],
        [0, -2, -10],  # distance=-1, even with large z
        [0, 1, 3],
        [0, -2, 0],
    ])
    plane = np.array([1, 1, 0])
    point = np.array([0, -1, 0])
    sign = 1
    expected_distances = np.array([0, 0, 0, 1, 3, -1, 2, -1])
    distances = phantom_base.distance_point_from_plane(points, plane, point, sign)
    np.testing.assert_array_equal(distances, expected_distances)

    # test the sign flip
    plane = np.array([1, 1, 0])
    point = np.array([0, -1, 0])
    # want this point to have a positive distance from the plane, but it's on the negative side
    test_point = [0, -2, 0]
    sign = np.sign((test_point - point).dot(plane))  # gives -1
    points = np.array([
        [0, -1, 0],
        [0, -2, 0],
        [1, 1, 1]
    ])
    expected_distances = np.array([0, 1, -3])
    distances = phantom_base.distance_point_from_plane(points, plane, point, sign)
    np.testing.assert_array_equal(distances, expected_distances)

    # test floats
    points = np.array([
        [0.5, 0.5, -0.5],
        [0.0, 0.0, 0.0],
        [0.5, -0.5, 1],
        [-0.1, 2, 6],
    ])
    point = np.array([0.5, 0.5, -0.5])
    sign = 1
    distances = phantom_base.distance_point_from_plane(points, plane, point, sign)
    expected_distances = np.array([0, -1, -1, 0.9])
    np.testing.assert_array_equal(distances, expected_distances)


def test_distance_point_from_point():
    # test accurate distances between a point and a set of points
    points = np.array([
        [0, -1, 0],  # will be new origin
        [0, -1, 1],
        [0, -1, -5],
        [0, 0, 0],
        [0, -2, 0],
    ])
    point = np.array([0, -1, 0])
    distances = phantom_base.distance_point_from_point(points, point)
    expected_distances = np.array([0, 1, 5, 1, 1])
    np.testing.assert_array_equal(distances, expected_distances)

    # test floats
    points = np.array([
        [0.5, 0.5, -0.5],
        [0.0, 0.0, 0.0],
        [0.5, -0.5, 1],
        [-0.1, 2, 6],
    ])
    point = np.array([0.5, 0.5, -0.5])
    distances = phantom_base.distance_point_from_point(points, point)
    expected_distances = np.array([0, 0.8660, 1.8027, 6.6977])
    np.testing.assert_array_almost_equal(distances, expected_distances, decimal=3)
