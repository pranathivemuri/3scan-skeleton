#!/usr/bin/env python3

import numpy as np
import skimage.morphology
import scipy.ndimage

import skeletonization.skeleton.thin_volume as thin_volume
import skeletonization.metrics.skeleton_graph_stats as skeleton_stats
import skeletonization.skeleton.vessel_phantom as vessel_phantom
import skeletonization.skeleton.phantom_noise as noise

CUBE_EDGE = 256
CYLINDER_RADIUS = 5
DECIMATION_FACTOR_Z = 5
TARGET_NAMES = ["background", "foreground"]
BASE_PATH = "/home/pranathi/pipeline_skeleton_results/sfn_phantoms_2017"

# To validate use -
# http://stim.ee.uh.edu/resources/software/netmets/
# Github code - https://git.stim.ee.uh.edu/segmentation/netmets
# Build guide - http://stim.ee.uh.edu/education/software-build-guide/
# usage - netmets objfile1 objfile2 --sigma 3


def draw_3d_line(x0, y0, z0, x1, y1, z1, array):
    """
    Draw Bresenhams line in the array given starting and ending point
    """
    # 'steep' xy Line, make longest delta x plane
    swap_xy = abs(y1 - y0) > abs(x1 - x0)
    if swap_xy:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    # do same for xz
    swap_xz = abs(z1 - z0) > abs(x1 - x0)
    if swap_xz:
        x0, z0 = z0, x0
        x1, z1 = z1, x1

    # delta is Length in each plane
    delta_x = abs(x1 - x0)
    delta_y = abs(y1 - y0)
    delta_z = abs(z1 - z0)

    # drift controls when to step in 'shallow' planes
    # starting value keeps Line centered

    drift_xy = (delta_x / 2)
    drift_xz = (delta_x / 2)

    # direction of line
    step_x = 1
    if (x0 > x1):
        step_x = -1
    step_y = 1
    if (y0 > y1):
        step_y = -1
    step_z = 1
    if (z0 > z1):
        step_z = -1

    # starting point
    y = y0
    z = z0
    points = []

    # step through longest delta (which we have swapped to x)
    for x in range(x0, x1, step_x):
        # copy position
        cx = x
        cy = y
        cz = z

        # unswap (in reverse)
        if swap_xz:
            cx, cz = cz, cx
        if swap_xy:
            cx, cy = cy, cx

        # passes through this point
        points.append((cx, cy, cz))
        if tuple((cx, cy, cz)) >= (0, 0, 0) and tuple((cx, cy, cz)) < array.shape:
            array[cx, cy, cz] = 1

        # update progress in other planes
        drift_xy = drift_xy - delta_y
        drift_xz = drift_xz - delta_z

        # step in y plane
        if (drift_xy < 0):
            y = y + step_y
            drift_xy = drift_xy + delta_x
        # same in z
        if (drift_xz < 0):
            z = z + step_z
            drift_xz = drift_xz + delta_x
    return array


def get_obj_write(skeletonized_arr, obj_path):
    # GET OBJS FILE TO RUN NETMETS
    ss = skeleton_stats.SkeletonStats(skeletonized_arr)
    metrics_results, obj_lines = ss.get_stats_general(ss.networkx_graph)
    obj_file = open(obj_path, "w")  # open a obj file in the given path
    obj_file.writelines(obj_lines)
    obj_file.close()


def get_ground_truth(cylinders):
    # DOESN'T PROPERLY CONNECT CYLINDERS AT BRANCHES NEEDS TUNING
    gt_skeleton = np.zeros(shape, dtype=bool)
    for cyl1, cyl2, _ in cylinders:
        gt_skeleton = draw_3d_line(
            cyl1[0], cyl1[1], cyl1[2], cyl2[0], cyl2[1], cyl2[2], gt_skeleton)
    return gt_skeleton


def save_thinning_results(phantom, cylinders, save_string):
    gt_skeleton = get_ground_truth(cylinders)
    gt_save_path = BASE_PATH + save_string + "skeleton_gt_mask"
    pr_save_path = BASE_PATH + save_string + "skeleton_pr_mask"
    py_save_path = BASE_PATH + save_string + "skeleton_py_mask"
    skeleton_python = skimage.morphology.skeletonize_3d(phantom)
    skeleton_pranathi = thin_volume.get_thinned(phantom, mode='reflect')
    np.save(BASE_PATH + save_string + "vessel_mask", phantom)
    np.save(gt_save_path, gt_skeleton)
    np.save(pr_save_path, skeleton_pranathi)
    np.save(py_save_path, skeleton_python)
    get_obj_write(gt_skeleton, gt_save_path + ".obj")
    get_obj_write(skeleton_pranathi, pr_save_path + ".obj")
    get_obj_write(skeleton_python.astype(bool), py_save_path + ".obj")


shape = (CUBE_EDGE, CUBE_EDGE, CUBE_EDGE)
# X CYLINDER
cylinder_in_xaxis, cylindersx = (vessel_phantom.x_cylinder(
    radius=CYLINDER_RADIUS, shape=shape) / 255).astype(np.bool)
save_thinning_results(cylinder_in_xaxis, cylindersx, "x_")
# Y CYLINDER
cylinder_in_yaxis, cylindersy = vessel_phantom.y_cylinder(
    radius=CYLINDER_RADIUS, shape=shape)
save_thinning_results(cylinder_in_yaxis, cylindersy, "y_")

# Z CYLINDER
cylinder_in_zaxis, cylindersz = vessel_phantom.z_cylinder(
    radius=CYLINDER_RADIUS, shape=shape)
save_thinning_results(cylinder_in_zaxis, cylindersz, "z_")

# DIAGONAL
cylinder_diagonal = vessel_phantom.diagonal(CUBE_EDGE, radius=CYLINDER_RADIUS)
save_thinning_results(
    cylinder_diagonal, [(0, 0, 0), (CUBE_EDGE - 1, CUBE_EDGE - 1, CUBE_EDGE - 1)], "xyz_")


# 3 VERTICAL CYLINDERS
vertical_cylinders, cylinders_vertical = vessel_phantom.vertical_cylinders(shape[0], shape[2])
save_thinning_results(vertical_cylinders, cylinders_vertical, "v_")

# NOISY 3 VERTICAL CYLINDERS
noisy_vertical_cylinders = noise.realistic_filter(vertical_cylinders, random_seed=42)
save_thinning_results(noisy_vertical_cylinders, cylinders_vertical, "nv_")


# VESSEL TREE
phantom_mask_tree = vessel_phantom.vessel_tree(CUBE_EDGE)
cylinders_tree = vessel_phantom.cylinders_base_to_pixelcoord(
    vessel_phantom.VESSEL_TREE_BASECOORD, (shape))
save_thinning_results(phantom_mask_tree, cylinders_tree, "t_")

# VESSEL TREE NOISY
noisy_tree = noise.realistic_filter(phantom_mask_tree, random_seed=42)
save_thinning_results(noisy_tree, cylinders_tree, "nt_")

# VESSEL LOOP
phantom_mask_loop = vessel_phantom.vessel_loop(CUBE_EDGE)
cylinders_loop = vessel_phantom.cylinders_base_to_pixelcoord(
    vessel_phantom.VESSEL_LOOP_BASECOORD, (shape))
save_thinning_results(phantom_mask_loop, cylinders_loop, "l_")

# VESSEL LOOP NOISY
noisy_loop = noise.realistic_filter(phantom_mask_loop, random_seed=42)
save_thinning_results(noisy_loop, cylinders_loop, "nl_")

# ----------------------------
# Decimation results to emulate anisotropic voxels of KESM images

# VESSEL TREE DECIMATED IN Z
phantom_mask_tree_decimated = scipy.ndimage.zoom(
    phantom_mask_tree, zoom=[1, 1, 1 / DECIMATION_FACTOR_Z], order=0)
volume_shape = (CUBE_EDGE, CUBE_EDGE, CUBE_EDGE / DECIMATION_FACTOR_Z)
cylinders_tree_decimated = vessel_phantom.cylinders_base_to_pixelcoord(
    vessel_phantom.VESSEL_TREE_BASECOORD, (volume_shape))
save_thinning_results(phantom_mask_tree_decimated, cylinders_tree_decimated, "td_")

# VESSEL TREE NOISY DECIMATED IN Z
noisy_decimated_tree = noise.realistic_filter(phantom_mask_tree_decimated, random_seed=42)
save_thinning_results(noisy_decimated_tree, cylinders_tree_decimated, "ntd_")

# VESSEL LOOP DECIMATED IN Z
phantom_mask_loop_decimated = scipy.ndimage.zoom(
    phantom_mask_loop, zoom=[1, 1, 1 / DECIMATION_FACTOR_Z], order=0)
cylinders_loop_decimated = vessel_phantom.cylinders_base_to_pixelcoord(
    vessel_phantom.VESSEL_LOOP_BASECOORD, (volume_shape))
save_thinning_results(phantom_mask_loop_decimated, cylinders_loop_decimated, "ld_")


# VESSEL LOOP NOISY DECIMATED IN Z
noisy_decimated_loop = noise.realistic_filter(phantom_mask_loop_decimated, random_seed=42)
save_thinning_results(noisy_decimated_loop, cylinders_loop_decimated, "nld_")
