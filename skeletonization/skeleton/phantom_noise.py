import numbers
import numpy as np

import skeleton.image_tools as image_tools


def realistic_filter(
        stack: np.array,
        foreground: int=40,
        background: int=200,
        edge_filter_sigma: int=7,
        speckle_level: int=20,
        gaussian_noise_sigma: int=5,
        random_seed: int=1) -> np.array:
    """
    Add "realistic" noise to the binary vessel representation
    Inverts the binary mask so that vessels are now dark and background is light,
    matching the expected output of india ink stained vessels

    Input parameters:
        stack : ndarray, binary mask, expecting ndim==3, may work for 4
        foreground : int, within stack dtype, value for foreground
        background : int, within stack dtype, value for background
        edge_filter_sigma : stdev of gaussian to blur the binmask edges
        speckle_level : range +/- to add pixel noise
        gaussian_noise_sigma : standard deviation of pixel-level gaussian noise
        random_seed : int, used to seed the random number generator for consistent results

    Returns:
        ndarray with same shape and dtype as stack, with brightfield noise applied

    NOTE: The expected noise distribution from microscopy data is Poisson. This
        does not use Poisson noise, but looks pretty good.
    """
    np.random.seed(random_seed)
    # squish the signal into more realistic bounds
    noise_stack = scale_binarymask_update(
        stack.copy(), foreground=foreground, background=background)
    # blur the edges of the vessels
    noise_stack = blur_volume(noise_stack, sigma=edge_filter_sigma)
    # add blurred uniform noise to volume
    noise_stack = add_speckle(
        noise_stack, level=speckle_level, sigma=5, random_seed=random_seed)
    # add gussian noise to volume
    noise_stack = add_gaussian_noise(
        noise_stack, sigma=gaussian_noise_sigma, random_seed=random_seed)
    return noise_stack.astype(np.uint8)


def scale_binarymask_update(
        stack: np.array,
        foreground: int=40,
        background: int=200) -> np.array:
    """
    rescale a binary mask (two classes, identified as min and max)
    to the given foreground and background values.

    Enforces that the input array consists of 1 or 2 values
        eg (0, 255), or (0, 1), or (30, 233).
    If only 1 value is found, this method sets all corresponding
        elements as the background
    NOTE: This could be a problem if we want the object to be foreground

    Inputs:
        stack : ndarray, two values present
        foreground: signal value, ie vessels, must be within dtype of stack
        background: background value, must be within dtype of stack

    Returns: stack updated in place with new value mapping

    Raises AssertionError if more than two unique values are found, or stack is empty
    """
    # find number of unique values. If empty or more than two, raise error
    unique_vals = np.unique(stack)
    nvals = len(unique_vals)
    assert nvals <= 2 and nvals != 0, \
        "1 or 2 unique values required, found {}".format(nvals)

    # get min and max before updating in place
    mmin = unique_vals.min()
    mmax = unique_vals.max()

    # find min value, set as background.
    # the min will be the only value if the volume is singular (only one value)
    stack[stack == mmin] = background

    # find max value and set as foreground.
    if mmin != mmax:
        stack[stack == mmax] = foreground

    return stack


def blur_volume(
        stack: np.array,
        sigma: int=21,
        out=None) -> np.array:
    """
    smooth out the edges of the vessels in 2D
    Uses a 2D kernel to blur on each face, with size=`sigma`

    Inputs :
        stack, 3D array of dtype of np.uint8 or float
        sigma : standard deviation of the Gaussian kernel used, in both (x, y)
        out : None, return allocated array
              ndarray, same shape as stack, to save output to.
    Returns :
        blurred volume, allocated

    NOTE: blur_volume() only works on types restricted within cv2.GaussianBlur(),
        which appears to be undocumented.
        uint8 and float64 work, int16 or int32 does not.
    NOTE: blur_volume does not blur in 3 dimensions, only across 2.
    TODO: create 3D gaussian blur; add 1D blur across Z
    """
    assert sigma > 0
    assert sigma % 2 != 0

    if out is None:
        out = np.empty_like(stack)
    else:
        np.testing.assert_array_equals(out.shape, stack.shape)
    for i in range(stack.shape[2]):
        out[:, :, i] = image_tools.gaussian_blur_2d(stack[:, :, i], sigma)
    return out


def add_gaussian_noise(
        stack: np.array,
        sigma: int=5,
        random_seed: int=1) -> np.array:
    """
    adds gaussian noise to the input stack, with standard deviation `sigma`
    """
    np.random.seed(random_seed)
    gnoise = (sigma * np.random.randn(*stack.shape))  # returns as float64
    return image_tools.round_float_uint8(stack + gnoise)


def add_speckle(
        stack: np.array,
        level: int=10,
        sigma: int=5,
        random_seed: int=1) -> np.array:
    """
    Adds multiple layers of noise from uniform distribution
    1) create layer of uniform noise defined by level (+- value, within 255)
    2) Filter layer with gaussian kernel with sigma (std of gaussian kernel)
    3) add back to original stack, and return as uint8

    Inputs:
        stack : ndrray
        sigma : int, stdev of gaussian kernel used in first convolution; only odd values allowed
        out : if not None, puts output in this array, asserts that has same shape as stack

    Return:
        new ndarray of the volume with "speckle" applied, same shape as stack

    Raises AssertionError if input `stack` is not np.uint8
    Raises AssertionError if sigma is not odd

    TODO: add in-place operation, with 'out' parameter
    """
    assert sigma > 0
    assert sigma % 2 != 0

    np.random.seed(random_seed)
    noise = np.random.randint(0, level * 2, stack.shape, dtype=np.uint8)
    blur = blur_volume(noise, sigma=sigma)  # blur the first level

    # we need to boost the dtype to signed to avoid twos complement rollover in uint8
    # When .astype() is used, it reallocates the array in memory, so should be used sparingly
    # Unfortunately, blur_volume() only works on types restricted within cv2.GaussianBlur(),
    # which is undocumented. uint8 and float64 work, int16 or int32 do not.
    out = stack.astype(np.int16) + blur - level  # shift back down to center on distribution

    return image_tools.round_float_uint8(out)


def add_stripe_noise(
        im: np.array,
        noise_std: numbers.Number=10,
        random_seed: numbers.Integral=None) -> np.array:
    """
    Create zero mean random normal noise that's constant in the vertical
    direction and adds the vertical stripes to the image.

    Inputs:
        im =          2D image, uint8; can be boxed or unboxed
        noise_std =   Standard deviation from which to draw random noise.
        random_seed = Seed for random number generation

    Returns:
        tuple: (image, color_stripe)
            image: ndarray, (M, N, 3)
            color_stripe: ndarray, (N, 3) of stripe data used

    Note: You might want to choose a variance that works with the images you're working with.
    If your images are close to zero you should lower the default noise_std.
    """
    # Seed random number generator
    np.random.seed(random_seed)
    # Create random vector across columns and channels from desired variance distribution
    column_stripes = noise_std * np.random.randn(*((1,) + im.shape[1:]))
    # Add noise to image
    im_striped = im + column_stripes

    # drop the first dimension of stripes; will need to readd it for correct array broadcasting
    return image_tools.round_float_uint8(im_striped), column_stripes.squeeze(axis=0)


def add_stripe_noise_on_stack(
        stack: np.array,
        noise_std: numbers.Number=10,
        random_seed: numbers.Integral=None) -> np.array:
    """
    Create zero mean random normal noise that's constant in the vertical
    direction. Adds the noise to each image in the stack.

    Inputs:
    im =          NPVolume stack, uint8
    noise_std =   Standard deviation from which to draw random noise
    random_seed = Seed for random number generation

    Outputs:
    noise_stack = Image stack with added noise
    """

    # Get z shape
    z = stack.shape[2]
    # We want different seeds for each z, but consistent because they're from the same initial seed
    np.random.seed(random_seed)
    z_seeds = np.random.randint(0, 1000, size=z)
    noise_stack = np.zeros_like(stack)
    for i in range(0, z):
        noise_stack[:, :, i, ...], stripes = add_stripe_noise(
            stack[:, :, i, ...],
            noise_std=noise_std,
            random_seed=z_seeds[i])

    return noise_stack
