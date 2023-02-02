import math

import numpy as np
import SimpleITK as sitk

from generate_sample_utils import elastic_deformation, is_iter, force_len, match_len, rescale, pad_to_same, get_smoothed_contour


# from fc numerical - updated to remove dependencies on fc
# https://github.com/norok2/flyingcircus_numeric
def coord(
        shape,
        position=0.5,
        is_relative=True):
    xx = force_len(position, len(shape))
    if is_relative:
        xx = tuple(rescale(x, 0, d - 1, 0, 1) for x, d in zip(xx, shape))
    return xx


# https://github.com/norok2/flyingcircus_numeric
def grid_coord(
        shape,
        position=0.5,
        is_relative=True,
        dense=False):
    xx0 = coord(shape, position, is_relative)
    grid = tuple(slice(-x0, dim - x0) for x0, dim in zip(xx0, shape))
    return np.ogrid[grid] if not dense else np.mgrid[grid]


# from raster_geometry - updated to remove dependencies on fc and fcn
# https://github.com/norok2/raster_geometry
def nd_superellipsoid(
        shape,
        semisizes=0.5,  # radius, so 0.5 is to the edge from center
        indexes=2,
        position=0.5,
        ndim=None,
        rel_position=True,
        rel_sizes=True,
        smoothing=False):
    """Generate a superellipsoid in n-dimensions.

    Parameters
    ----------
    shape : int | tuple[int, ...]
        Either a single integer or a tuple with an integer for each dimension of the output image
    semisizes : float | tuple[float, ...], optional
        Radius of the output ellipsoid, by default 0.5
    position : float | tuple[float, ...], optional
        Centerpoint of the output ellipsoid in the output image, by default 0.5
    ndim : int, optional
        The number of output dimensions, by default None
    rel_position : bool, optional
        Whether to use the position argument as a percentage of the output shape, by default True
    rel_sizes : bool, optional
        Whether to use the semisizes argument as a percentage of the output shape, by default True
    smoothing : bool | float, optional
        The amount of smoothing to apply to the output (or the boolean of whether to apply it), by default False

    Returns
    -------
    numpy.ndarray
        An array of the superellipsoid
    """
    if ndim is None:
        ndim = max([len(item) if is_iter(item) else 1 for item in (shape, position, semisizes, indexes)])

    # check compatibility of given parameters
    # shape = gouda.force_len(shape, ndim)
    # position = gouda.force_len(position, ndim)
    # semisizes = gouda.force_len(semisizes, ndim)
    # indexes = gouda.force_len(indexes, ndim)
    (shape, position, semisize, indexes), _ = match_len(shape, position, semisizes, indexes, count=ndim)

    # get correct position
    semisizes = coord(shape, semisizes, is_relative=rel_sizes)  # gets center coord of shape, only needs shape for ndim and if size is relative
    xx = grid_coord(shape, position, is_relative=rel_position)

    rendered = np.zeros(shape, dtype=float)
    for x_i, semisize, index in zip(xx, semisizes, indexes):
        rendered += (np.abs(x_i / semisize) ** index)
    if smoothing is False:
        rendered = rendered <= 1.0
    else:
        if smoothing > 0:
            k = math.prod(semisizes) ** (1 / index / ndim / smoothing)
            rendered = np.clip(1.0 - rendered, 0.0, 1.0 / k) * k
        else:
            rendered = rendered.astype(float)
    return rendered


def make_sphere(sphere_radius, seed=None):
    """Create a sphere image

    Parameters
    ----------
    sphere_radius : int | tuple[int, int]
        Either a defined radius or the range [min, max) for a random radius to be sampled from
    seed : numpy.random.Generator | Any, optional
        Either a random generator or the seed to create a new random generator, by default None

    Returns
    -------
    sitk.Image
        The SimpleITK image of the sphere
    """
    if isinstance(seed, np.random.Generator):
        random = seed
    else:
        random = np.random.default_rng(seed)

    if is_iter(sphere_radius):
        sphere_radius = random.integers(*sphere_radius)
    sphere = nd_superellipsoid(sphere_radius * 2, semisizes=0.5, ndim=3)
    sphere_img = sitk.GetImageFromArray(sphere.copy().astype(np.uint8))
    return sphere_img, sphere_radius


def create_nodule(core_sphere_radius=(30, 50), aux_sphere_radius=(20, 30), num_spheres=6, shift_type='avgRadius',
                deform=True, sigma=2, alpha=2, smooth_iter=20, smooth_pass_band=0.01, seed=None):
    """Create a nodule image

    Parameters
    ----------
    core_sphere_radius : int | tuple[int, int], optional
        Either a defined radius or the range [min, max) for a random radius to be sampled from, by default (30, 50)
    aux_sphere_radius : int | tuple[int, int], optional
        Either a defined radius or the range [min, max) for a random radius to be sampled from, by default (20, 30)
    num_spheres : int, optional
        The number of auxiliary spheres to generate, by default 6
    shift_type : str, optional
        The way to pick the distance auxiliary spheres are placed from the main sphere, by default 'avgRadius'
    deform : bool, optional
        Whether to apply elastic deformation to the combined nodule, by default True
    sigma : int, optional
        The sigma for elastic deformation, by default 2
    alpha : int, optional
        The alpha for elastic deformation, by default 2
    smooth_iter : int, optional
        The number of iterations for the smoothing, by default 20
    smooth_pass_band : float, optional
        The pass band for the smoothing, by default 0.01
    seed : np.random.Generator | Any, optional
        Either a random generator or a seed for a random generator, by default None

    Returns
    -------
    sitk.Image
        An image of the nodule of combined spheres
    """
    if isinstance(seed, np.random.Generator):
        random = seed
    else:
        random = np.random.default_rng(seed)

    sphere_img, radius = make_sphere(core_sphere_radius, seed=random)
    spheres = []
    radii = []
    upper_shifts = []
    lower_shifts = []

    for _ in range(num_spheres):
        sphere_img2, radius2 = make_sphere(aux_sphere_radius, seed=random)
        spheres.append(sphere_img2)
        radii.append(radius2)
        if isinstance(shift_type, (int, float)):
            shift_size = shift_type
        elif shift_type == 'minRadius':
            shift_size = min(radius, radius2)
        elif shift_type == 'maxRadius':
            shift_size = max(radius, radius2)
        elif shift_type == 'avgRadius':
            shift_size = (radius + radius2) / 2
        shift = (random.random(size=3) * 2) - 1
        shift = np.ceil((shift / np.linalg.norm(shift)) * shift_size).astype(int)
        upper_shifts.append(np.maximum(shift, 0))
        lower_shifts.append(np.abs(np.minimum(shift, 0)))

    spheres = pad_to_same(sphere_img, *spheres, share_info=True)

    global_upper = np.max(upper_shifts, axis=0)
    global_lower = np.max(lower_shifts, axis=0)

    sphere_img = sitk.ConstantPad(spheres[0], global_upper.tolist(), global_lower.tolist())
    # sphere_img = spheres[0].apply(sitk.ConstantPad, )
    sphere_img.SetOrigin([0, 0, 0])
    spheres = spheres[1:]

    for idx in range(len(spheres)):
        sphere = spheres[idx]
        upper_shift = upper_shifts[idx]
        lower_shift = lower_shifts[idx]

        upper_pad = (global_upper - upper_shift) + lower_shift
        lower_pad = (global_lower - lower_shift) + upper_shift
        sphere = sitk.ConstantPad(sphere, upper_pad.tolist(), lower_pad.tolist())
        # sphere = sphere.apply(sitk.ConstantPad, upper_pad.tolist(), lower_pad.tolist(), in_place=False)
        sphere.SetOrigin([0, 0, 0])
        sphere_img = sphere_img | sphere

    sphere_img = get_smoothed_contour(sphere_img, num_iterations=smooth_iter, pass_band=smooth_pass_band)
    if deform:
        sphere_img = elastic_deformation(sphere_img, sigma=sigma, alpha=alpha)
    return sphere_img


if __name__ == '__main__':
    for i in range(10):
        test = create_nodule()
        sitk.WriteImage(test, f"noduleImage_{i:02d}.nii.gz")
