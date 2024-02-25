from nilearn import image, datasets
from nilearn.maskers import NiftiMasker
from nibabel import Nifti1Image
import numpy as np
from .coordinates import Coordinate2mm

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

def mask_by_significance(img1, significance_img, alpha=0.05, mask=None):
    """Apply a significance mask to an image.

    Masks the input image (`img1`) based on the significance levels
    provided in another image (`significance_img`). Pixels in `img1` 
    corresponding to non-significant regions in `significance_img` 
    (above the significance threshold) are set to zero.

    Parameters:
    -----------
    img1 : Niimg-like object
        Input image to be masked.

    significance_img : Niimg-like object
        Image containing significance levels. Pixels with values 
        above the significance threshold (1 - alpha) are considered 
        significant.

    alpha : float, optional
        Significance level (default is 0.05).

    mask : Niimg-like object, optional
        Masking image. If provided, only pixels within the mask 
        will be considered for masking (default is None).

    Returns:
    --------
    masked_img : Nifti1Image
        Masked image with non-significant regions set to zero.
    """
    try:
        img1 = apply_mni_mask(img1)
        significance_img = apply_mni_mask(significance_img)
    except Exception as e:
        print(f'Error: {e}')
        return None
    masked_img = image.math_img(f"img1 * (img2 > {1-alpha})", img1=img1, img2=significance_img)
    return masked_img

def return_peak_coordinate(nii_img, apply_anatomical_mask=True):
    if type(nii_img) == str:
        nii_img = image.load_img(nii_img)
    elif not isinstance(nii_img, Nifti1Image):
        raise ValueError('Input must be a path to a NIfTI image or a NIfTI image object')
    if apply_anatomical_mask:
        nii_img = apply_mni_mask(nii_img)
    peak_index = np.unravel_index(np.argmax(nii_img.get_fdata()), nii_img.shape)
    peak_index = tuple(int(x) for x in peak_index)
    peak_coord = Coordinate2mm(peak_index, 'voxel').mni_space_coord
    return peak_coord

