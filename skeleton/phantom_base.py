import numbers
import warnings

import numpy as np

"""
Pixel coordinate conversion

The phantom_base.py module creates raster volumes from vector
representations using CELL_BASED coordinates. That is, each pixel is referred to
by the coordinate of the pixel center.
+---+---
| x |
+---+-
|   |

where the center `x` is numerically represented as (0, 0), and
the top left `+` is (-0.5, -0.5). This cell-based coordinate system is useful
when the volume is numerically represented between [-1, 1] across all cube dimensions.

Arrays are typically numerically represented in EDGE-BASED coordinates, where
the top left `+` is numerically represented as (0, 0) and the center
`x` is (0.5, 0.5).

If a volume's raster shape is (32, 32, 32), we will want to refer to a point in
that shape via pixel coordinate. The coordinate system used is undefined, i.e.
does the pixel (31, 31, 31) map to (1, 1, 1), or 1-eps (where eps is the raster
conversion factor)?
"""
CAP_TYPES = ["flat", "sphere", "none"]


def to_basecoord(x: int, shape: int):
    """
    Scale a single coordinate from pixel-based edge coordinate to numerical
    cell-based coordinate for vector-to-raster.

    Inputs:
        x -- integer coordinate, must be < shape
        shape -- the size of the dimension we are scaling from

    Output:
        the coordinate in base [-1, 1] coordinate space, to be used directly with
        phantom_base methods

    Raises:
        AssertionError when x is not within range [0, shape)

    Eg:
    >>> s = 32  # the shape of the raster dimension we are scaling to
    >>> x = 0   # the coordinate of the pixel we are scaling
    >>> to_basecoord(x, s)
    -1.0
    >>> to_basecoord(31, s)
    1.0
    """
    assert shape > 0, "Shape must be positive integer, found {}".format(shape)
    assert 0 <= x < shape, "point `{}` is not within shape `{}`".format(x, shape)
    if shape == 1:
        shape = 2  # make sure that we can still have a single dimension and dont divide by zero
    return -1 + (x * 2) / (shape - 1)


def scale_point_to_basecoord(point: (int, int, int), shape: (int, int, int)):
    """
    Take an n-tuple `point` within a volume's `shape`, and scale to coordinate in a -1:1 unit volume
    """
    return tuple(to_basecoord(p, s) for p, s in zip(point, shape))


def scale_radius_to_basecoord(radius: int, shape: tuple):
    """
    Scale a radius from pixel space to base space [-1, 1]

    We divide by the minimum of the XY dimension. Since we may want to have a single z
    face, we scale by the minimum of the XY coordinates, which are also likely to be
    the same shape.

    Inputs:
        radius -- integer radius
        shape -- the size of the dimension we are scaling from

    Returns:
        radius scaled to base coordinate space, between [-1, 1],
            scaled by the minimum of the XY shape

    Raises an AssertionError if radius is a negative value
    """
    assert radius > 0, "Radius must be a positive integer, given {}".format(radius)
    return (radius * 2) / min(shape[:2])


def to_pixelcoord(base: float, shape: int):
    """
    Scale the coordinate from base coordinate (in -1:1 unit volume) to pixel coordinate (0:shape)
    """
    assert -1.0 <= base <= 1.0, "point base dimension coordinate is not within [-1, 1]"
    return int(np.round((base + 1) * (shape - 1) / 2))


def scale_point_to_pixelcoord(point: (float, float, float), shape: (int, int, int)):
    """
    Scale a 3-tuple point to its coordinate in pixel space defined by `shape`
    """
    return tuple(to_pixelcoord(p, s) for p, s in zip(point, shape))


def scale_radius_to_pixelcoord(radius: float, shape: tuple):
    """
    Scale the cylinder radius from base space [-1, 1] to pixel space
    We scale by the minimum of the XY dimension, as in `scale_radius_to_basecoord`
    """
    assert radius > 0.0, "Radius must be positive, given {}".format(radius)
    return int(np.round((radius * min(shape[:2])) / 2))


def create_coordinate_block(shape: (int, int, int)):
    """
    Create three blocks where the indices key to the
    normalized (-1 to 1) coordinate in three space.

    This can be used to construct more complex objects
    in three dimensions.

    Keyword Arguments:
        shape {tuple} -- Shape of the coordinate grid
    """
    return np.mgrid[-1.0:1.0:1.0j * shape[0],
                    -1.0:1.0:1.0j * shape[1],
                    -1.0:1.0:1.0j * shape[2]]


def trace_function(xyzr_callable,
                   shape: (int, int, int)=(256, 256, 256),
                   isotropy: tuple=(1, 1, 1),
                   step_count: int=100):
    '''
    This is a tool to turn a parameterized function into a
    3d raster feature.  It outputs an array with shape specified
    where points "inside" the feature are masked to 255.  All other
    values will be 0.

    xyzr_callable is a function which must take a single floating
    point argument, and produce a tuple with 4 floating point elements
    the elements are representative of a ball (x, y, z, r) which will
    be considered as 'within' the raster.

    xyzr_callable will be called "step_count" times with values ranging
    from 0 to 1.  This allows the user to select their perferred parametric
    mapping for output.

    This way complex rasters can be constructed from simple parametric
    functions.  This is used as a workhorse in several simple phantoms
    generated below.

    Input parameters:
        xyzr_callable: a method object that takes in a value t between [0, 1],
            and returns (x, y, z, r)

        shape: a 3-tuple defining the shape of the returned raster (voxel) volume

        isotropy: a 3-tuple that indicates the scaling per dimension
            Since the parameterization is created in a normalized space [-1, 1],
            any non-cubic volumes will need to be scaled appropriately to have
            the correct sphere radius in that direction. That is, if isotropy
            isn't adjusted correctly for a non-cubic volume, a sphere would become
            a squashed ellipsoid.

            To appropriately scale isotropy, the target axis isotropy value
            should be scaled by the size of the target axis shape and
            the unary axis shape
            shape = (64, 64, 128)
            isotropy = (1, 1, 0.5) == (1, 1, shape[0] / shape[2]))

        step_count: how many steps to iterate across the parameterization

    Returns:
        ndarray, dtype uint8, of input shape. Values are 0 and 255.

    '''
    output = np.zeros(shape, dtype=np.uint8)
    xi, yi, zi = create_coordinate_block(shape)

    # Accrue the steps first, and unique them
    for t in np.linspace(0.0, 1.0, step_count):
        xc, yc, zc, r = xyzr_callable(t)
        mask = np.sqrt((xc - xi) ** 2 / isotropy[0] ** 2 +
                       (yc - yi) ** 2 / isotropy[1] ** 2 +
                       (zc - zi) ** 2 / isotropy[2] ** 2) < r

        nd_spacing = 1.0 / min(shape)
        if r < nd_spacing:
            warnings.warn("Radius of phantom is approaching grid rez. {} v. {}".format(r, nd_spacing))
        output[mask] = 255

    return output


def cylinder_in_x(r, shape: (int, int, int)=(256, 256, 256), step_count=512):
    def cylinder(t):
        return -1 + (t * 2), 0, 0, r

    return trace_function(cylinder, shape=shape, step_count=step_count)


def cylinder_in_xy(r, shape: (int, int, int)=(256, 256, 256), step_count=512):
    def cylinder(t):
        return -1 + (t * 2), -1 + (t * 2), 0, r

    return trace_function(cylinder, shape=shape, step_count=step_count)


def cylinder_in_xyz(r, shape: (int, int, int)=(256, 256, 256), step_count=512):
    def cylinder(t):
        return -1 + (t * 2), -1 + (t * 2), -1 + (t * 2), r

    return trace_function(cylinder, shape=shape, step_count=step_count)


def torus_xy(r1, r2, shape: (int, int, int)=(256, 256, 256), step_count=101):
    """
    Create a raster of a torus with radii r1, r2.
     - r1 is the small radius (donut cross section)
     - r2 is the large radius.

    (r2 - r1) is the center hole size
    """
    # NOTE(meawoppl) - This is technically valid,
    # but unlikey to be what is intended.
    assert r2 > r1, "Degenerate torus"

    # NOTE(meawoppl) even step count might fail to
    # come out with full D4 symmetry
    def torus(t):
        theta = 2 * np.pi * t
        return r2 * np.cos(theta), r2 * np.sin(theta), 0, r1

    return trace_function(torus, shape=shape, step_count=step_count)


def contiguous_elements(arr: np.array):
    """
    Finds contiguous True regions of the boolean array "arr".

    :param array: 1D boolean ndarray
    :returns array: 2d with first column as the (inclusive) start index of the matching
        regions, and second column is the (exclusive) end index.

    :Example:

    >>> kmath.contiguous_elements(np.array([0, 1, 1, 1, 2]) == 1)
    ... array([[1, 4]])

    """
    d = np.diff(arr)  # Find the indices of changes in "array"
    idx, = d.nonzero()
    # the start index is the next element after the change, so shift by 1
    idx += 1

    # deal with edges of diff
    if arr[0]:  # If the start of arr is True prepend a 0
        idx = np.r_[0, idx]
    if arr[-1]:  # If the end of arr is True, append the length
        idx = np.r_[idx, arr.size]

    idx.shape = (-1, 2)  # Reshape the result into two columns
    return idx


def nonzero_bounds(arr: np.array):
    """
    Return a tuple of the indices (inclusive, exclusive) that bracket nonzero values of `arr`.

    Useful for identifying the max and min bins of a histogram,
    or rising and falling edges of an object

    :param arr: 1D ndarray
    :returns tuple: first and last index of non-zero values.
        The last index is one greater than the last nonzero value (exclusive)

    :raises ValueError: if no nonzero elements found
    :raises AssertionError: if not a 1D array
    """
    np.testing.assert_array_equal(arr.ndim, 1)
    nonzeros = np.nonzero(arr)[0]
    if nonzeros.size == 0:
        raise ValueError("No nonzero elements found in array")
    return min(nonzeros), max(nonzeros) + 1


def normalized(arr: np.array, axis: int=-1, order: int=2):
    """
    Normalize vector using 2-norm, across arbitrary axes,

    Deals with length 0 vectors well; avoids dividing by zero

    :param arr: ndarray, will be normalized along given `axis`
    :param axis: axis to normalize against (default=-1), same guidelines as np.linalg.norm
    :param order: the order of the normalization factor, default L2 (default=2)
        follows same guidelines as np.linalg.norm

    :returns: ndarray of same dimensions as `arr`
    """
    norm_length = np.atleast_1d(np.linalg.norm(arr, order, axis))
    # set any zero elements to be 1, avoiding divide by zero errors
    norm_length[norm_length == 0] = 1
    if arr.ndim == 1:
        # single dimensions are speyshul
        return arr / norm_length
    return arr / np.expand_dims(norm_length, axis)  # use array broadcasting


def cylinder(shape,
             p1: (numbers.Number, numbers.Number, numbers.Number),
             p2: (numbers.Number, numbers.Number, numbers.Number),
             r: numbers.Number,
             cap="flat",
             dtype=np.uint8):
    """
    Create a boolean right circular cylinder between two points, p1 and p2, with radius r

    Inputs:
        shape: shape of the volume to allocate
        p1: 3-tuple with coordinates of endpoint of cylinder, in base coordinates [-1, 1]
        p2: 3-tuple with coordinates of endpoint of cylinder, in base coordinates [-1, 1]
        r: uniform radius of the cylinder, in base coordinates
            Anything larger than 2 will be outside of the shape of the volume
        cap: str, or None.
            Use one of the types available in base.CAP_TYPES (Default="flat")
            "flat" -- cuts the cylinder at each cylinder point (p1 & p2), leaving
                    a circular plane cut normal to the line
                    between (p2 - p1), at points p1 and p2. The resulting plane
                    should have radius `r`.
            "sphere" -- adds a spherical end at each of the points given.
                    The sphere will have the same radius as the cylinder at that point.
            "none" -- Leaves a cylinder along the given line between the two points, but
                    does not terminate at the point; it is fully expanded to fill
                    the line across the full volume

        foreground: the value to use in voxels inside the cylinder, default=1
        dtype: data type of the returned array, default=np.uint8

    Returns:
        ndarray, boolean, shape=`shape`

    Displays warnings.warn if radius is close to grid resolution
    Displays warnings.warn if radius is larger than the volume grid (>2)
    Raises AssertionError if cap is not in base.CAP_TYPES
    """
    assert cap in CAP_TYPES
    nd_spacing = 1.0 / min(shape)
    if r < nd_spacing:
        warnings.warn(
            "Radius of phantom is approaching grid rez. {} v. {}".format(r, nd_spacing))
    if r >= 2:
        warnings.warn(
            "Radius of cylinder is larger than the base volume: {} >= 2".format(r))
    # allocate the output array
    output = np.zeros(shape, dtype=bool)

    # get the volume's coordinates in R^3 [-1, 1] inclusive
    xi, yi, zi = create_coordinate_block(shape)
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    # wrap input tuples as arrays
    p1 = np.array(p1)
    p2 = np.array(p2)

    # first create the infinite cylinder
    cyl_length = np.linalg.norm(p2 - p1)  # length of cylinder

    # This is essentially the cross product of the difference between a given point
    # p and the two points that define our line, squared and normalized
    # distances_vec ?= np.cross((points - p1), (points - p2)) ** 2 / (cyl_length ** 2)
    # This gives us the distance from all points to our cylinder center line
    # We aren't using the full vectorized form yet because we may need to have
    # a space to put in the isotropy vector for each component, and I wanted
    # to confirm this method works first.
    # We will somehow need to add isotropy to this equation
    distances = (((yi - y1) * (zi - z2) - (zi - z1) * (yi - y2)) ** 2 +
                 ((zi - z1) * (xi - x2) - (xi - x1) * (zi - z2)) ** 2 +
                 ((xi - x1) * (yi - y2) - (yi - y1) * (xi - x2)) ** 2) / (cyl_length ** 2)
    output[distances <= r ** 2] = 1
    # we have an infinite cylinder in our volume now

    # create set of vectorized points
    points = np.vstack((xi, yi, zi)).reshape(3, -1).T
    # Set the plane equation as the normalized vector, defined by the line between p2 and p1,
    # with p1 as the reference point
    normal = normalized(p2 - p1)

    # Find the correct sign by seeing which side the other point is on
    # sign will be zero if the two points are the same
    # We use p1 as reference, since thats what we did with the `normal` above
    sign = np.sign((p2 - p1).dot(normal))

    if cap != "none":
        # always trim the infinite cylinder here
        # any use of points could be masked to only the vectors that are already on
        # this would make the math a lot faster.
        # would need to figure out how to reallocate back into output array

        # first point
        plane_distance = distance_point_from_plane(points, normal, p1, -sign).reshape(shape)
        output[plane_distance > 0] = 0
        # second point
        plane_distance = distance_point_from_plane(points, normal, p2, sign).reshape(shape)
        output[plane_distance > 0] = 0

    if cap == "sphere":
        # requires that we have already cropped to "flat"
        # add a sphere on the cap after we chop the infinite cylinder
        sphere_distance = distance_point_from_point(points, p1).reshape(shape)
        output[sphere_distance < r] = 1
        sphere_distance = distance_point_from_point(points, p2).reshape(shape)
        output[sphere_distance < r] = 1

    return output


def distance_point_from_plane(points: np.array,
                              plane_normal: np.array,
                              ref_point: np.array,
                              sign=1):
    """
    Returns distance of all input points from a given plane

    Inputs:
        points: NxM ndarray, with N points of dimension M
            eg: a (200, 3) array of 200 3D point coordinates
        plane_normal: A length M 1D ndarray that defines the normal of a plane
            (a, b, c) would define the plane at ax + by + cz = 0
        ref_point: A length M ndarray, the point we use to offset the plane from the origin
        sign: integer, decides the negative or positive value of the returned distance
            One way to get the target sign is to Find a point that is on the side
            of the plane that you want, and use the sign of that point to decide
            the correct distance sign for the volume you want to mask.
            Default=1, or
            A positive `sign` will give positive distances on the associated target side
            You can get the correct target sign via:
            >>> target_sign = np.sign((selection_point - ref_point).dot(plane_normal))

    Returns:
        Nx1 ndarray of the distance from the input vectors to the plane

    Raises AssertionError if sign is anything other than -1 or 1

    TODO: sanity check that these are actual vector distances
    This doesnt matter when we are just using the sign, but it does if we actually want distance
    """
    assert sign == 1 or sign == -1, \
        "'sign' must be positive or negative 1, got {}".format(sign)
    return sign * (points - ref_point).dot(plane_normal)


def distance_point_from_point(points: np.array,
                              ref_point: np.array):
    """
    Distance of all input points from a given point
    Can be used to create a spherical mask around a point

    Inputs:
        points: NxM ndarray, with N points of dimension M
            eg: a (200, 3) array of 200 3D point coordinates
        ref_point: A length M ndarray, the relative origin

    Returns:
        Nx1 ndarray of the distance from the input vectors to the point
    """
    return np.linalg.norm(points - ref_point, axis=-1)
