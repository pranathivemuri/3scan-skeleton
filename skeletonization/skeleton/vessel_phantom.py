import numpy as np

import skeletonization.skeleton.phantom_base as phantom_base


"""
Test library to create 3D raster volume of vessels
default settings use a light background with dark vessels, mimicking
the output of india-ink perfused brightfield microscopy images

ex:
>>> bstack = vessel_tree()
>>> stack = realistic_filter(bstack)

Using mayavi is a useful way to visualize these test stacks
Install mayavi via:
    conda install -c menpo mayavi=4.5.0
This version works on python 3. Versions < 4.5 don't work on python 3.x
mayavi also requires that QT4 (pyqt) is installed. This may be in conflict
with matplotlib >=1.5.3, which started to use qt5

View the stack contours:
>>> import mayavi as mlab
>>> mlab.contour3d(stack, colormap='gray')


This library is the start of taking vectorized inputs and creating raster
representations of that data in a large block (possibly gigabytes) in memory.
Use caution in allocating large volumes, especially in larger data types.

Todo: import VTK as dependency, use to create primitives and create mesh
booleans over a ndarray raster volume

As of 2016.01.10, mayavi 4.4.x works in python 3, but only with pyqt4. Newer
matplotlib (>=1.5.3) defaults to pyqt5 and qt5. It is difficult to have an
environment that supports both dependencies.
mayavi requires a newer version of VTK as well.
See https://github.com/enthought/mayavi/issues/84 for more info
Still unclear which branch of VTK works on python 3. Often requires special builds
The 'menpo' anaconda package releases may work better than most
"""


# base coordinates were created with rescale vessel vectors in 0 to 512 space to -1 to 1
VESSEL_LOOP_BASECOORD = [
    ((-0.980, -0.941, -1.000), (0.879, 0.566, 1.000), 0.137),
    ((-0.397, -0.460, -0.374), (-0.632, 0.487, 0.566), 0.078),
    ((-0.523, 0.010, 0.104), (-0.006, 0.777, 0.796), 0.066),
    ((0.096, 0.080, 0.840), (0.272, 0.080, -0.843), 0.066),
    ((-0.084, 0.640, 0.679), (0.249, 0.065, -0.577), 0.066)
]

VESSEL_TREE_BASECOORD = [
    ((-0.980, -0.941, -1.000), (0.879, 0.566, 1.000), 0.137),
    ((-0.397, -0.460, -0.374), (-0.632, 0.487, 0.566), 0.078),
    ((-0.523, 0.057, 0.143), (-0.022, 0.566, 0.957), 0.059),
    ((-0.049, -0.276, -0.057), (0.659, -0.256, 0.859), 0.047),
    ((0.405, 0.217, 0.440), (-0.186, 0.186, -0.217), 0.047),
    ((0.245, 0.096, 0.350), (0.260, 0.796, 0.996), 0.070),
    ((-0.589, -0.644, 1.000), (0.746, -0.217, -1.000), 0.059),
    ((-0.663, 0.487, -0.804), (-0.863, -0.883, 1.000), 0.078),
    ((-0.730, -0.382, 0.272), (-0.100, -0.609, 1.000), 0.059)
]

VESSEL_LOOP_512 = [
    ((5, 15, 0), (480, 400, 511), 35),
    ((154, 138, 160), (94, 380, 400), 20),
    ((122, 258, 282), (254, 454, 459), 17),
    ((280, 276, 470), (325, 276, 40), 17),
    ((234, 419, 429), (319, 272, 108), 17),
]


VESSEL_TREE_512 = [
    ((5, 15, 0), (480, 400, 511), 35),
    ((154, 138, 160), (94, 380, 400), 20),
    ((122, 270, 292), (250, 400, 500), 15),
    ((243, 185, 241), (424, 190, 475), 12),
    ((359, 311, 368), (208, 303, 200), 12),
    ((318, 280, 345), (322, 459, 510), 18),
    ((105, 91, 511), (446, 200, 0), 15),
    ((86, 380, 50), (35, 30, 511), 20),
    ((69, 158, 325), (230, 100, 511), 15)
]


def cylinder_on_axis(radius: int=5, axis: int=0, shape: (int, int, int)=(256, 256, 256)):
    """
    create cylinder with `radius` in pixels along center of `axis` of 3D volume
    Allocates and returns ndarray of given `shape` [default: [256, 256, 256])

    radius is normalized to the minimum dimension of the volume shape

    Inputs:
        radius: int, cylinder radius in pixels
        axis: int, which axis the cylinder will be parallel to (eg, 0 = x axis)
        shape: tuple, the shape of the volume to render the cylinder in

    Returns:
        ndarray of binary np.uint8 with `shape`, unique values [0, 1]
    """
    assert axis < len(shape), "Invalid axis index specified, must be < len(shape)"
    r = phantom_base.scale_radius_to_basecoord(radius, shape)

    def cylinder_axis(t):
        u = -1 + (t * 2), 0, 0
        # roll the elements to have the formula on the target axis
        return tuple(np.roll(u, axis)) + (r,)

    # TODO: we need to scale step_count appropriately for the input shape; maybe half?
    nsteps = max(shape)
    return phantom_base.trace_function(cylinder_axis, shape=shape, step_count=nsteps)


def add_cylinder_px(volume,
                    point1: (int, int, int),
                    point2: (int, int, int),
                    radius: int):
    """
    Given two points and a radius in pixel space, allocate a volume with the target cylinder
    Internally, radius is normalized to the minimum dimension of the volume shape

    Inputs:
        volume: ndarray, grayscale volume (unboxed)
        point1: tuple, describes starting point of cylinder in volume, in volume coordinates
            eg, (0, 64, 256)
        point2: tuple, describes starting point of cylinder in volume, in volume coordinates
        radius: int, the cylinder radius in pixels
            Internally, radius is normalized to the minimum dimension of the volume shape


    Return:
        ndarray, dtype=bool, same shape as input `volume`
    """
    p1, p2, r = cylinders_px_to_basecoord([(point1, point2, radius)], volume.shape)[0]
    return add_cylinder_basecoord(volume, p1, p2, r)


def add_cylinder_basecoord(
        volume,
        point1: (float, float, float),
        point2: (float, float, float),
        radius: float,
        cap="sphere"):
    """
    Given two points and a radius in pixel space, allocate a volume with the target cylinder
    Internally, radius is normalized to the minimum dimension of the volume shape

    Inputs:
        volume: ndarray, grayscale volume (unboxed)
        point1: tuple, describes starting point of cylinder in volume, in base volume floats [-1, 1]
            eg, (-1, 0.56, 0.98)
        point2: tuple, describes starting point of cylinder in volume, in base volume floats [-1, 1]
        radius: float, the cylinder radius in base volume coordinates
        cap: type of cap on the cylinder, eg "flat", "sphere", or None.
            Types are defined in phantoms.base.CAP_TYPES

    Return:
        ndarray, dtype=bool, same shape as input `volume`

    TODO: Make this smarter:
        1) only allocate volume for minimum bounding box for the cylinder we are adding
        2) update original input `volume` slice with allocated cylinder volume
        Even better, mkae a cylinder as a primitive, instead of making cylinder out of spheres
    """
    # TODO add isotropy back to base cylinder call
    # # scale the isotropy
    # shape_min = min(volume.shape)  # used as scaling factor
    # isotropy = tuple(shape_min / s for s in volume.shape)

    # A smarter way to do this would be to allocate the minimum bounding box
    # necessary to enclose the cylinder, and insert that volume into a slice
    # of the original volume. Hard part is getting the scale units correct.
    newvolume = phantom_base.cylinder(volume.shape, point1, point2, radius,
                                      cap=cap)
    return np.logical_or(volume, newvolume)


def volume_bool_to_dtype(volume, fg=255, dtype=np.uint8):
    """
    Convert boolean volume to have a scaled volume as dtype

    Inputs:
        volume -- NPVolume
        fg -- foreground value, [default=255]
        dtype -- numpy data type, [default=np.uint8]

    Returns:
        NPVolume, with True scaled to fg
    """
    if dtype is bool:
        return volume
    return volume.astype(dtype) * fg


def cylinders_px_to_basecoord(cylinder_list: list, volume_shape: (int, int, int)):
    """
    Convert list of cylinder tuples in pixel coordinates to float coordinates, [-1, 1] inclusive
    """
    cylinders_base_list = []
    for point1, point2, radius in cylinder_list:
        p1 = phantom_base.scale_point_to_basecoord(point1, volume_shape)
        p2 = phantom_base.scale_point_to_basecoord(point2, volume_shape)
        r = phantom_base.scale_radius_to_basecoord(radius, volume_shape)
        cylinder = p1, p2, r
        cylinders_base_list.append(cylinder)

    return cylinders_base_list


def cylinders_base_to_pixelcoord(cylinder_list: list, volume_shape: (int, int, int)):
    """
    Convert list of cylinder tuples in base coordinates to pixel coordinates
    NOTE: we are not scaling the radius here
    """
    cylinders_px_list = []
    for point1, point2, radius in cylinder_list:
        p1 = phantom_base.scale_point_to_pixelcoord(point1, volume_shape)
        p2 = phantom_base.scale_point_to_pixelcoord(point2, volume_shape)
        r = phantom_base.scale_radius_to_pixelcoord(radius, volume_shape)
        cylinders_px_list.append((p1, p2, r))
    return cylinders_px_list


def create_cylinders_volume(
        shape: (int, int, int),
        cylinders_list: list,
        foreground=1,
        dtype=np.uint8):
    """
    Create a volume of given shape, with cylinders drawn in it

    Inputs:
        shape: tuple, the shape of the target volume in voxel
        cylinders: list of cylinder tuples, where each cylinder is represented as:
            ((x1, y1, z1), (x2, y2, z2), radius)
            Coordinates are in voxel space, not "base" coordinates.
        foreground: value to use as foreground for each voxel, default=1
        dtype: output array data type, default np.uint8

    Returns:
        Allocated binary volume of given `shape`, with cylinders drawn as true
    """
    volume = np.zeros(shape, dtype=bool)
    for point1, point2, radius in cylinders_list:
        volume = add_cylinder_px(volume, point1, point2, radius)
    return volume_bool_to_dtype(volume, fg=foreground, dtype=dtype)


def diagonal(cube_edge: int=128,
             radius: int=10,
             foreground: int=1,
             dtype=np.uint8):
    """
    Create a simple cylinder going from the origin to the far corner
    Inputs:
        cube_edge : int, the length of the cube along each edge
        radius : int, radius of the cylinder about the line
            from (0, 0, 0) to (cube_edge, cube_edge, cube_edge)
        foreground : int, vessel fill level, within dtype (default=1)
        dtype : data type to allocate
    Returns:
        ndarray of type `dtype`
    """
    if 2 * radius > cube_edge:
        raise ValueError("Given radius '{}' is larger than than cube edge length {}"
                         .format(radius, cube_edge))
    stack = np.zeros((cube_edge, cube_edge, cube_edge), dtype=bool)
    cylinder = [
        ((0, 0, 0), (cube_edge - 1, cube_edge - 1, cube_edge - 1), radius)
    ]
    stack = add_cylinder_px(stack, *cylinder[0])
    return volume_bool_to_dtype(stack, fg=foreground, dtype=dtype)


def z_cylinder(shape: (int, int, int), radius: int, dtype=np.uint8):
    """
    Create a vessel phantom of 1 cylinder along the z axis, in the middle of the volume

    Inputs:
        shape: tuple of ints, giving the allocated volume shape
        radius: radius of the cylinder
        dtype: data type, default=np.uint8

    Returns:
        tuple:
            (ndarray of type `dtype`, list of cylinders used)
    """
    z_depth = shape[2]
    xhalf_atom = shape[0] // 2
    yhalf_atom = shape[1] // 2
    cylinders = [
        ((xhalf_atom, yhalf_atom, 0),
            (xhalf_atom, yhalf_atom, z_depth - 1),
            radius)
    ]
    data_mask = create_cylinders_volume(shape, cylinders, foreground=1, dtype=dtype)
    return data_mask, cylinders


def vertical_cylinders(xy_size: int, z_depth: int, dtype=np.uint8):
    """
    Create a vessel phantom of 3 cylinders oriented along the z axis
    Typical use case is to divide this volume into a 3x3, for use in pipeline testing

    Inputs:
        xy_size: size of one side of the XY image
        z_depth: size of the z dimension
        dtype: data type, default=np.uint8

    Returns:
        tuple:
            (ndarray of type `dtype`, list of cylinders used)
    """
    shape = (xy_size, xy_size, z_depth)
    image_size_px = shape[0] // 3
    z_depth = shape[2]
    half_atom = image_size_px // 2
    quarter_atom = image_size_px // 4
    cylinders = [
        # center cylinder, z-aligned, 64x64 radius = 16
        ((image_size_px + half_atom, image_size_px + half_atom, 0),
            (image_size_px + half_atom, image_size_px + half_atom, z_depth - 1),
            image_size_px // 4),
        # first tile overlapping to other tiles, z-aligned, 64x64 radius = 16
        ((image_size_px - quarter_atom, image_size_px - quarter_atom, 0),
            (image_size_px - quarter_atom, image_size_px - quarter_atom, z_depth - 1),
            image_size_px // 4),
        # lower middle tile overlapping to other tiles, z-aligned, 64x64 radius = 8
        ((image_size_px * 2 + quarter_atom, image_size_px + half_atom, 0),
            (image_size_px * 2 + quarter_atom, image_size_px + half_atom, z_depth - 1),
            image_size_px // 8),
    ]
    data_mask = create_cylinders_volume(shape, cylinders, foreground=1, dtype=dtype)
    return data_mask, cylinders


def vessel_loop(cube_edge: int=128, foreground: int=1):
    """
    Creates a volume with cylinders connected to form an arbitrary polygonal loop
    Uses VESSEL_LOOP_512 for the cylinder coordinates

    Inputs:
        cube_edge : int, the length of the cube along each edge
        foreground : int, vessel fill level, within dtype (default=1)

    Returns:
        ndarray of shape (cube_edge, cube_edge, cube_edge) and dtype=np.uint8
    """
    shape = (cube_edge, cube_edge, cube_edge)
    volume = np.zeros(shape, dtype=np.uint8)
    cylinder_list_float = cylinders_px_to_basecoord(VESSEL_LOOP_512, (512, 512, 512))

    for point1, point2, radius in cylinder_list_float:
        volume = add_cylinder_basecoord(volume, point1, point2, radius)

    return volume_bool_to_dtype(volume, fg=foreground, dtype=np.uint8)


def vessel_tree(cube_edge: int=128, foreground: int=1):
    """
    Creates a volume with cylinders connected to form an arbitrary polygonal loop
    Uses VESSEL_TREE_512 for the cylinder coordinates

    Inputs:
        cube_edge : int, the length of the cube along each edge
        foreground : int, vessel fill level, within dtype (default=1)

    Returns:
        ndarray of shape (cube_edge, cube_edge, cube_edge) and dtype=np.uint8
    """
    shape = (cube_edge, cube_edge, cube_edge)
    volume = np.zeros(shape, dtype=np.uint8)
    cylinder_list_float = cylinders_px_to_basecoord(VESSEL_TREE_512, (512, 512, 512))

    for point1, point2, radius in cylinder_list_float:
        volume = add_cylinder_basecoord(volume, point1, point2, radius)

    return volume_bool_to_dtype(volume, fg=foreground, dtype=np.uint8)
