from nilearn import image, datasets
from nilearn.maskers import NiftiMasker
from nibabel import Nifti1Image
import numpy as np
from .coordinates import Coordinate2mm
import os
from scipy import ndimage as ndi
# from scipy.ndimage import zoom, measurements
from scipy.stats import rankdata
import pandas as pd

current_dir = os.path.dirname(__file__)
atlases_dir = os.path.join(current_dir, 'atlas_data')
mni_mask_path = os.path.join(atlases_dir, 'mni152_2mm_brain_mask.nii.gz')

def get_mni_mask():
    """Return the MNI152 brain mask as a NIfTI image."""
    return image.load_img(mni_mask_path)

def apply_mni_mask_fast(path_or_nifti_img):
    """
    Apply MNI mask to the input NIfTI image.

    Parameters:
    path_or_nifti_img : str or Nifti1Image
        If str, it represents the path to the NIfTI image file.
        If Nifti1Image, it represents the NIfTI image object.

    Returns:
    Nifti1Image
        Masked NIfTI image.

    Raises:
    ValueError: If input is neither a path to a NIfTI image nor a NIfTI image object.
    """
    if isinstance(path_or_nifti_img, (str, Nifti1Image)):
        nifti_img = image.load_img(path_or_nifti_img) if isinstance(path_or_nifti_img, str) else path_or_nifti_img
    else:
        raise ValueError('Input must be a path to a NIfTI image or a NIfTI image object')

    mni_mask = image.load_img(mni_mask_path)
    masked_data = nifti_img.get_fdata() * mni_mask.get_fdata()
    masked_img = image.new_img_like(nifti_img, masked_data, affine=nifti_img.affine, copy_header=True)
    return masked_img

def apply_mni_mask(path_or_nifti_img):
    """
    Apply the MNI152 brain mask to a NIfTI image and return the masked image
    with the extra dimension squeezed out. The function handles both file paths
    and Nifti1Image objects as input. The MNI152 mask is resampled to match the
    target image's resolution and affine before masking. The resulting masked
    image retains the spatial information and metadata of the original image.

    Parameters
    ----------
    path_or_nifti_img : str or image.image.Nifti1Image
        The input NIfTI image. This can either be a file path to a NIfTI file or
        a Nifti1Image object. 

    Returns
    -------
    squeezed_img_3d : image.image.Nifti1Image
        The masked NIfTI image with the extra (singleton) dimension removed,
        ensuring the output is a 3D Nifti1Image. This image has been masked
        with the MNI152 template, resampled to match the original image's
        resolution and affine, and retains the header information of the
        original image.

    Raises
    ------
    ValueError
        If the input is neither a string path nor a Nifti1Image object.

    Example
    -------
    >>> masked_img = apply_mni_mask('path/to/nifti_file.nii')
    Or
    >>> from nilearn import image
    >>> nifti_img = image.load_img('path/to/nifti_file.nii')
    >>> masked_img = apply_mni_mask(nifti_img)
    
    masked_img can now be used for further analysis or visualization.
    """
    if type(path_or_nifti_img) == str:
        nifti_img = image.load_img(path_or_nifti_img)
    elif isinstance(path_or_nifti_img, Nifti1Image):
        nifti_img = path_or_nifti_img
    else:
        raise ValueError('Input must be a path to a NIfTI image or a NIfTI image object')
    brain_mask = datasets.load_mni152_brain_mask()
    resampled_brain_mask = image.resample_to_img(source_img=brain_mask, target_img=nifti_img, interpolation='nearest')
    masker = NiftiMasker(mask_img=resampled_brain_mask, standardize=False)
    masker.fit()
    masked_data = masker.transform(nifti_img)
    masked_img_3d = masker.inverse_transform(masked_data)
    squeezed_data = masked_img_3d.get_fdata()[:, :, :, 0]
    squeezed_img_3d = image.new_img_like(masked_img_3d, squeezed_data, affine=masked_img_3d.affine, copy_header=True)
    return squeezed_img_3d

def apply_custom_mask(img1, masking_img, threshold=0.95):
    """Apply a another mask to an image, such as a significance mask.

    Masks the input image (`img1`) based on the significance levels
    provided in another image (`masking_img`). Pixels in `img1` 
    corresponding to non-significant regions in `masking_img` 
    (above the significance threshold) are set to zero.

    Parameters:
    -----------
    img1 : Niimg-like object
        Input image to be masked.

    masking_img : Niimg-like object
        Image containing significance levels. Pixels with values 
        above the threshold are considered 

    treshold : float, optional
        Significance level (default is 0.95).

    Returns:
    --------
    masked_img : Nifti1Image
        Masked image with non-significant regions set to zero.
    """
    try:
        img1 = apply_mni_mask_fast(img1)
        masking_img = apply_mni_mask_fast(masking_img)
    except Exception as e:
        print(f'Error: {e}')
        return None
    masked_img = image.math_img(f"img1 * (img2 > {threshold})", img1=img1, img2=masking_img)
    return masked_img

def return_peak_coordinate(nii_img, apply_anatomical_mask=True):
    """Return the peak coordinate of a NIfTI image.

    If the input is a string, it's assumed to be a file path to a NIfTI image,
    which will be loaded. Otherwise, it's expected to be a NIfTI image object.

    Parameters:
    -----------
    nii_img : str or Nifti1Image
        Path to a NIfTI image file or a NIfTI image object.

    apply_anatomical_mask : bool, optional
        Whether to apply an anatomical mask to the input image (default is True).

    Returns:
    --------
    peak_coord : tuple
        MNI coordinates of the peak value in the image.
    """

    # Load NIfTI image if input is a file path
    if isinstance(nii_img, str):
        nii_img = image.load_img(nii_img)
    elif not isinstance(nii_img, Nifti1Image):
        raise ValueError('Input must be a path to a NIfTI image or a NIfTI image object')

    # Apply anatomical mask if specified
    if apply_anatomical_mask:
        nii_img = apply_mni_mask_fast(nii_img)

    # Find the peak index
    peak_index = np.unravel_index(np.argmax(nii_img.get_fdata()), nii_img.shape)
    peak_index = tuple(int(x) for x in peak_index)

    # Convert peak index to MNI coordinates
    peak_coord = Coordinate2mm(peak_index, 'voxel').mni_space_coord

    return peak_coord

def anatomical_label_at_idx(idx, coord_space='mni'):
    """Get the anatomical label at the specified coordinates.

    Parameters:
    -----------
    idx : tuple, list, or numpy array
        Coordinates of the point of interest.

    coord_space : str, optional
        Coordinate space of the input coordinates (default is 'mni').

    Returns:
    --------
    anatomical_label : str
        Anatomical label corresponding to the specified coordinates.
    """

    # Convert idx to tuple if it's a list or numpy array
    if isinstance(idx, list):
        idx = tuple(idx)
    elif isinstance(idx, np.ndarray):
        idx = tuple(idx.tolist())

    # Ensure idx is a tuple
    if not isinstance(idx, tuple):
        raise ValueError("idx must be a tuple, list, or numpy array.")

    anatomical_label = Coordinate2mm(idx, coord_space).anatomical_name
    return anatomical_label

def find_local_maxima(img, order=10):
    """Detects local maxima in a 3D array

    Parameters
    ---------
    img : str to Nifti1Image or Nifti1Image
    order : int
        How many points on each side to use for the comparison

    Returns
    -------
    coords : ndarray
        coordinates of the local maxima
    values : ndarray
        values of the local maxima
    """
    if isinstance(img, str):
        img = image.load_img(img)
    elif not isinstance(img, Nifti1Image):
        raise ValueError('Input must be a path to a NIfTI image or a NIfTI image object')
    data = img.get_fdata()
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data, footprint=footprint)
    mask_local_maxima = data > filtered
    coords = np.asarray(np.where(mask_local_maxima)).T # What does this look like?
    coords = [tuple(x) for x in coords]
    coords = [Coordinate2mm(x, 'voxel').mni_space_coord for x in coords]
    values = data[mask_local_maxima]
    df = pd.DataFrame({'coord': coords, 'value': values})
    try:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.dropna(subset=['value'], inplace=True)
        df = df.sort_values(by='value', ascending=False, ignore_index=True)
    except Exception as e:
        print(f"Error building DataFrame: {e}")
        display(df)
        
    return df

def unpack_img_list(img_list):
    """Unpacks a list of NIfTI images into a list of 3D NIfTI images.

    Parameters:
    -----------
    img_list : list
        List of NIfTI images.

    Returns:
    --------
    unpacked_img_list : list
        List of 3D NIfTI images.
    """
    unpacked_img_list = [apply_mni_mask_fast(img) for img in img_list]
    return unpacked_img_list

def stack_img_list(img_list):
    """Stacks a list of 3D NIfTI images into a 4D NIfTI image.

    Parameters:
    -----------
    img_list : list
        List of 3D NIfTI images.

    Returns:
    --------
    stacked_img : Nifti1Image
        4D NIfTI image.
    """
    img_data = [img.get_fdata() for img in img_list]
    stacked_data = np.vstack(img_data, axis=-1)
    return stacked_data

def correlate_img_with_list(img, img_list):
    if type(img) == str:
        img = image.load_img(img)
    elif not isinstance(img, Nifti1Image):
        raise ValueError('Input must be a path to a NIfTI image or a NIfTI image object')
    img = apply_mni_mask_fast(img)
    img_list = unpack_img_list(img_list)

    # Correlate the input image with each image in the list
    img_data = img.get_fdata()
    img_list_data = [img.get_fdata() for img in img_list]
    correlations = [np.corrcoef(img_data.ravel(), img_list_data[i].ravel())[0, 1] for i in range(len(img_list_data))]
    return correlations

def normalize_to_quantile(img):
    """
    Transforms the values of a NIfTI object into their corresponding quantile scores, 
    effectively normalizing the distribution of values.
    Args:
        img (Nifti1Image, path to such an image): The input NIfTI image to be normalized.

    Raises:
        ValueError: If there's a shape mismatch between the input and output data, indicating an error in 
            processing or data handling.

    Returns:
        img (Nifti1Image): The input NIfTI image with its values transformed to quantile scores.
    """
    img = apply_mni_mask_fast(img)
    nd_array = img.get_fdata()
    original_shape = nd_array.shape
    
    # Mask finite values
    finite_mask = np.isfinite(nd_array)

    # Flatten finite values and calculate quantile scores
    data_flat = nd_array[finite_mask].flatten()
    data_ranked = rankdata(data_flat)
    data_quantile_scores = data_ranked / len(data_ranked)

    # Initialize output array with NaNs and populate with quantile scores
    output_array = np.full_like(nd_array, np.nan, dtype=np.float64)
    output_array[finite_mask] = data_quantile_scores

    img = image.new_img_like(img, output_array, affine=img.affine, copy_header=True)
    if img.get_fdata().shape != original_shape:
        raise ValueError("Shape mismatch between input and output data")
    return img