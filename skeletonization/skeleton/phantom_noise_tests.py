import nose.tools
import numpy as np

import skeletonization.skeleton.phantom.noise as phantom_noise


def test_scale_binarymask_update():
    mask = np.random.randint(0, 1, size=(10, 10, 10))

    # test binary mask
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[2:4, 2:4, 2:4] = 1
    mask_scaled = phantom_noise.scale_binarymask_update(mask, foreground=20, background=200)
    np.testing.assert_array_equal(np.unique(mask_scaled),
                                  np.array([20, 200],
                                  dtype=np.uint8))

    # test one value
    mask = np.ones((5, 5, 5), dtype=np.uint8)
    mask_scaled = phantom_noise.scale_binarymask_update(mask, foreground=20, background=200)
    np.testing.assert_array_equal(np.unique(mask_scaled),
                                  np.array([200],
                                  dtype=np.uint8))

    # test 3 values
    grayscale = np.zeros((3, 3, 3), dtype=np.uint8)
    for i in range(3):
        grayscale[i, i, i] = i
    with nose.tools.assert_raises(AssertionError):
        mask_scaled = phantom_noise.scale_binarymask_update(grayscale)

    # test empty
    mask = []
    with nose.tools.assert_raises(AssertionError):
        mask_scaled = phantom_noise.scale_binarymask_update(mask)


def test_blur_volume():
    stack = np.zeros((6, 6, 2), dtype=np.uint8)
    stack[2:4, 2:4, :] = 255
    blur = phantom_noise.blur_volume(stack, sigma=3)
    np.testing.assert_array_equal(blur.shape, stack.shape)

    # count pixels per grayscale value to check blur accuracy
    counts, vals = np.histogram(blur, bins=range(256))
    found_values = vals[np.where(counts)[0]]
    np.testing.assert_array_equal(found_values, [0, 16, 48, 143])
    found_counts = counts[found_values]
    np.testing.assert_array_equal(found_counts, [40, 8, 16, 8])

    # check that both faces are blurred equally
    np.testing.assert_array_equal(blur[..., 0], blur[..., 1])

    # check the in-place operation
    phantom_noise.blur_volume(stack, sigma=3, out=stack)
    np.testing.assert_array_equal(blur, stack)

    with nose.tools.assert_raises(AssertionError):
        phantom_noise.blur_volume(np.ones((2, 2, 2), dtype=np.uint8), sigma=2)


def test_add_speckle():
    stack = np.ones((20, 20, 2), np.uint8) * 128
    noisy = phantom_noise.add_speckle(stack, level=15, sigma=5, random_seed=1)
    np.testing.assert_array_equal(noisy.shape, stack.shape)
    np.testing.assert_array_equal(noisy.min(), 120)
    np.testing.assert_array_equal(noisy.max(), 133)
    np.testing.assert_array_equal(noisy.mean(), 126.72499999999999)

    with nose.tools.assert_raises(AssertionError):
        phantom_noise.add_speckle(np.ones((2, 2, 2)))

    with nose.tools.assert_raises(AssertionError):
        phantom_noise.add_speckle(np.ones((2, 2, 2), dtype=np.uint8), sigma=2)


def test_add_gaussian_noise():
    stack = np.ones((2, 2, 2), np.uint8) * 128
    gnoise = phantom_noise.add_gaussian_noise(stack, sigma=5, random_seed=1)
    target = np.array(
        [[[136, 125],
          [125, 123]],
         [[132, 116],
          [137, 124]]],
        dtype=np.uint8)
    np.testing.assert_array_equal(gnoise.shape, stack.shape)
    np.testing.assert_array_equal(gnoise, target)


def test_add_stripe_noise():
    # Test that stripe noise introduces noise in the vertical direction
    # and that same random seed generates same noise image
    im = np.ones((100, 50, 1), dtype=np.uint8) * 128
    im_noise, stripes = phantom_noise.add_stripe_noise(im, random_seed=42)
    # Assert that noisy image returns correct shape and type
    np.testing.assert_array_equal(im_noise.shape, im.shape)
    # check that stripes shape matches image shape
    nose.tools.assert_equals(stripes.shape, im.shape[1:])
    # Assert that noise has no variance vertically and std near 10 horizontally
    np.testing.assert_array_equal(sum(np.var(im_noise, axis=0)), 0)
    nose.tools.assert_less_equal(np.mean(np.std(im_noise, axis=1)), 10 + 1)
    nose.tools.assert_greater_equal(np.mean(np.std(im_noise, axis=1)), 10 - 1)
    # Assert that same seed returns same noise
    im_noise2, stripes2 = phantom_noise.add_stripe_noise(im, random_seed=42)
    np.testing.assert_array_equal(im_noise, im_noise2,
                                  err_msg="Same seed should produce same output.")


def test_add_stripe_noise_squeezed():
    # Test that stripe noise introduces noise in the vertical direction
    # and that same random seed generates same noise image
    im = np.ones((100, 50), dtype=np.uint8) * 100
    im_noise, stripes = phantom_noise.add_stripe_noise(im, random_seed=42)
    # Assert that noisy image returns correct shape and type
    np.testing.assert_array_equal(im_noise.shape, im.shape)
    # check that stripes shape matches image shape
    nose.tools.assert_equals(stripes.shape, im.shape[1:])

    # Assert that noise has no variance vertically and std near 10 horizontally
    np.testing.assert_array_equal(sum(np.var(im_noise, axis=0)), 0)
    nose.tools.assert_less_equal(np.mean(np.std(im_noise, axis=1)), 10 + 1)
    nose.tools.assert_greater_equal(np.mean(np.std(im_noise, axis=1)), 10 - 1)
    # Assert that same seed returns same noise
    im_noise2, stripes2 = phantom_noise.add_stripe_noise(im, random_seed=42)
    np.testing.assert_array_equal(im_noise, im_noise2,
                                  err_msg="Same seed should produce same output.")


def test_add_stripe_noise_color():
    # Test that stripe noise introduces noise in the vertical direction
    # and that same random seed generates same noise image
    im = np.ones((100, 50, 3), dtype=np.uint8) * 128
    im_noise, stripes = phantom_noise.add_stripe_noise(im, random_seed=42)
    # Assert that noisy image returns correct shape and type
    np.testing.assert_array_equal(im_noise.shape, im.shape)
    # check that stripes shape matches image shape
    nose.tools.assert_equals(stripes.shape, im.shape[1:])

    # Assert that noise has no variance vertically and std near 10 horizontally
    np.testing.assert_array_equal(sum(np.var(im_noise, axis=0)), 0)
    nose.tools.assert_less_equal(np.mean(np.std(im_noise, axis=1)), 10 + 1)
    nose.tools.assert_greater_equal(np.mean(np.std(im_noise, axis=1)), 10 - 1)
    # Assert that same seed returns same noise
    im_noise2, stripes2 = phantom_noise.add_stripe_noise(im, random_seed=42)
    np.testing.assert_array_equal(im_noise, im_noise2,
                                  err_msg="Same seed should produce same output.")


def test_add_stripe_noise_on_stack():
    # Test that noise stack is correct shape and dtype,
    # and that same random seed generates same noise stack
    im_stack = np.ones((100, 50, 5, 3), dtype=np.uint8) * 128
    im_noise = phantom_noise.add_stripe_noise_on_stack(im_stack, random_seed=42)
    # Assert that noisy image returns correct shape and type
    np.testing.assert_array_equal(im_noise.shape, im_stack.shape)
    # Assert that same seed returns same noise
    im_noise2 = phantom_noise.add_stripe_noise_on_stack(im_stack, random_seed=42)
    np.testing.assert_array_equal(im_noise, im_noise2,
                                  err_msg="Same seed should produce same output.")
