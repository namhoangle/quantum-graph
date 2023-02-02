import functools

import itk
import numpy as np
import SimpleITK as sitk
import vtk


# from gouda - updated to remove typing in case Python<3.9
def force_len(x, count, pad='wrap'):
    """Force the length of x to a given count

    Parameters
    ----------
    x : Any
        The item/iterable to force to a length of count
    count : int
        The output tuple length
    pad : str, optional
        Padding method for extending x. Can be either 'wrap' or 'reflect', by default 'wrap'
    """
    if not is_iter(x):
        return (x, ) * count
    else:
        if len(x) == count:
            return x
        elif len(x) < count:
            if pad == 'wrap':
                result = list(x)
                while len(result) < count:
                    diff = count - len(result)
                    result.extend(x[:diff])
                return type(x)(result)
            elif pad == 'reflect':
                if len(x) * 2.0 < count:
                    raise ValueError('Cannot reflect enough to force length.')
                return tuple(list(x) + list(reversed(x))[:count - len(x)])
            else:
                raise ValueError(f'Unknown padding method: {pad}.')
        else:
            return x[:count]


# from gouda - updated to remove typing in case Python<3.9
def match_len(*args, count=None, pad='wrap'):
    """Force all input items to the same length

    Parameters
    ----------
    count : Optional[int], optional
        The length to set all items to. If None, uses the length of the longest item, by default None
    pad : str
        The padding to use to extend an item. Can be either 'wrap' or 'reflect', by default 'wrap'
    """
    if count is None:
        count = max([len(item) if is_iter(item) else 1 for item in args])
    return tuple(force_len(item, count, pad=pad) for item in args), count


# from gouda - updated to remove typing in case Python<3.9
def is_iter(x, non_iter=(str, bytes, bytearray)):
    """Check if x is iterable

    Parameters
    ----------
    x : Any
        The variable to check
    non_iter : Iterable[type], optional
        Types to not count as iterable types, by default (str, bytes, bytearray)
    """
    if isinstance(x, tuple(non_iter)):
        return False

    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True


# from gouda - updated to remove typing in case Python<3.9
def rescale(data, output_min=0, output_max=1, input_min=None, input_max=None, axis=None):
    """Rescales data to have range [new_min, new_max] along axis or axes indicated

    Parameters
    ----------
    data : npt.ArrayLike
        Input array-like to rescale
    output_min : float, optional
        The minimum output value, by default 0
    output_max : float, optional
        The maximum output value, by default 1
    input_min : float, optional
        The minimum input value, by default None (if None, inferred from data along axis)
    input_max : float, optional
        The maximum input value, by default None (if None, inferred from data along axis)
    axis : Optional[ShapeType], optional
        Axis or axes along which to infer input min/max if needed, by default None

    Returns
    -------
    FloatArrayType
        Rescaled array

    NOTE
    ----
    For flexibility, there is no checking that input_min and input_max are actually the minimum and maximum values in data along axis. If they are not, the output values are rescaled as if they were and may lie outside of [output_min, output_max]. For enforced bounds, use `gouda.data_methods.clip`.
    """
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(float)
    min_val = np.min(data, axis=axis, keepdims=True) if input_min is None else np.asarray(input_min)
    max_val = np.max(data, axis=axis, keepdims=True) if input_max is None else np.asarray(input_max)
    data_range = max_val - min_val  # If max_val < min_val, the output will have flipped signs
    x = np.divide(data - min_val, data_range, where=data_range != 0, out=np.zeros_like(data))
    new_range = output_max - output_min
    return (x * new_range) + output_min


# adapted from GoudaMI
def pad_to_same(*images, bg_val=0, upper_pad_only=True, share_info=False):
    """Pad all images to the same size.

    Parameters
    ----------
    *images : sitk.Image
        The images to pad
    bg_val : int, optional
        Background pad value, by default 0
    upper_pad_only : bool, optional
        Whether to only pad away from the origin
    share_info : bool, optional
        Whether to copy information from the first image across all images, by default False

    NOTE
    ----
    Only use share_info if you don't care about origin/spacing/direction of the images. Not all images get the same padding, so their origin will likely shift differently.
    """
    sizes = [np.array(image.GetSize()) for image in images]
    largest = np.max(sizes, axis=0)
    results = []
    shared_ref = None
    for image in images:
        image_size = np.array(image.GetSize())
        pad = largest - image_size
        upper_pad = (pad // 2).astype(int)
        lower_pad = (pad - upper_pad).astype(int)
        if upper_pad.sum() + lower_pad.sum() > 0:
            image = sitk.ConstantPad(image, lower_pad.tolist(), upper_pad.tolist(), constant=bg_val)
        if share_info:
            if shared_ref is None:
                shared_ref = image
            else:
                image.CopyInformation(shared_ref)
        results.append(image)
    return results


# adapted from GoudaMI
def sitk2itk(image):
    if isinstance(image, itk.Image):
        return image
    itk_image = itk.GetImageFromArray(sitk.GetArrayViewFromImage(image), is_vector=image.GetNumberOfComponentsPerPixel() > 1)
    itk_image.SetOrigin(image.GetOrigin())
    itk_image.SetSpacing(image.GetSpacing())
    itk_image.SetDirection(itk.GetMatrixFromArray(np.reshape(np.array(image.GetDirection()), [image.GetDimension()] * 2)))
    return itk_image


# adapted from GoudaMI
def itk2sitk(image):
    if isinstance(image, sitk.Image):
        return image
    sitk_image = sitk.GetImageFromArray(itk.GetArrayViewFromImage(image), isVector=image.GetNumberOfComponentsPerPixel() > 1)
    sitk_image.SetOrigin(tuple(image.GetOrigin()))
    sitk_image.SetSpacing(tuple(image.GetSpacing()))
    sitk_image.SetDirection(itk.GetArrayFromMatrix(image.GetDirection()).flatten())
    return sitk_image


# adapted from GoudaMI
def wrap4itk(func):
    """Wrap a method that takes a vtk.ImageData and returns a vtk.PolyData"""
    @functools.wraps(func)
    def wrapped_func(image, *args, **kwargs):
        input_type = ''
        input_direction = image.GetDirection()
        if isinstance(image, sitk.Image):
            image = sitk2itk(image)
            input_type = 'sitk'
        elif isinstance(image, itk.Image):
            input_type = 'itk'
        else:
            raise ValueError("Unknown input type: {}".format(type(image)))
        image = itk.vtk_image_from_image(image)
        result = func(image, *args, **kwargs)

        stencil = vtk.vtkPolyDataToImageStencil()
        stencil.SetInputData(result)
        stencil.SetInformationInput(image)
        stencil.Update()

        converter = vtk.vtkImageStencilToImage()
        converter.SetInputData(stencil.GetOutput())
        converter.SetInsideValue(1)
        converter.SetOutsideValue(0)
        converter.SetOutputScalarTypeToUnsignedChar()
        converter.Update()

        result = itk.image_from_vtk_image(converter.GetOutput())
        if input_type == 'sitk':
            result = itk2sitk(result)
        else:
            # Should be input_type == 'itk'
            pass
        result.SetDirection(input_direction)
        return result
    return wrapped_func


# adapted from GoudaMI
@wrap4itk
def get_smoothed_contour(contour, num_iterations=20, pass_band=0.01):
    """Apply the vtkWindowedSincPolyData Filter to a vtkImageData

    Parameters
    ----------
    contour : vtk.ImageData
        The contour to be smoothed
    num_iterations : int
        Number of iterations for the vtkWindowedSincPolyDataFilter
    pass_band : float
        The pass band for vtkWindowedSincPolyDataFilter

    Returns
    -------
    vtk.PolyData
    """
    discrete = vtk.vtkDiscreteFlyingEdges3D()
    discrete.SetInputData(contour)
    # discrete.GenerateValues(n, 1, n)
    discrete.GenerateValues(1, 1, 1)
    discrete.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(discrete.GetOutput())
    smoother.SetNumberOfIterations(num_iterations)
    smoother.SetPassBand(pass_band)
    smoother.Update()

    return smoother.GetOutput()


# adapted from GoudaMI
def elastic_deformation(image, sigma=1.0, alpha=2.0, interp=sitk.sitkLinear, seed=None):
    """Perform a random elastic deformation on the image.

    Parameters
    ----------
    image : ImageType
        Image to deform
    sigma : Union[float, tuple[float, ...]], optional
        The smoothness of the deformation - higher is smoother, by default 1.0
    alpha : Union[float, tuple[float, ...]], optional
        The magnitude of the deformation, by default 2.0
    seed : Optional[Union[np.random.Generator, int]], optional
        The seed or generator for random values, by default None

    NOTE
    ----
    sigma and alpha can either be single float values or tuples of values for each dimension
    """
    if isinstance(seed, np.random.Generator):
        random = seed
    else:
        random = np.random.default_rng(seed)
    deformation = random.random(size=(*image.GetSize()[::-1], image.GetDimension())) * 2 - 1
    def_image = sitk.GetImageFromArray(deformation, isVector=True)
    def_image.CopyInformation(image)
    def_image = sitk.SmoothingRecursiveGaussian(def_image, sigma=sigma)
    def_image = multiply_vector_image(def_image, alpha)

    warp_filt = sitk.WarpImageFilter()
    warp_filt.SetInterpolator(interp)
    warp_filt.SetOutputParameteresFromImage(image)
    return warp_filt.Execute(image, def_image)


# adapted from GoudaMI
def multiply_vector_image(image: sitk.Image, scalar):
    """Multiply a vector image by a scalar

    Parameters
    ----------
    image : sitk.Image
        vector image to multiply
    scalar : Union[float, Sequence[float]]
        The scalar or list of scalars to multiply by
    """
    if not is_iter(scalar):
        scalar = (scalar,) * image.GetNumberOfComponentsPerPixel()
    assert len(scalar) == image.GetNumberOfComponentsPerPixel(), "Scalar must be a single value or a sequence of values the same length as the number of components in the image"
    return sitk.Compose([sitk.VectorIndexSelectionCast(image, i) * scalar[i % len(scalar)] for i in range(image.GetNumberOfComponentsPerPixel())])
