from nilearn.datasets import *
from nilearn import image as nli
import numpy as np
import nibabel as nib

def load_mni152_brain_maskX(resolution=None, threshold=0.2):
    """
    Note from nilearnX: The same as nilearn.datasets.load_mni152_brain_mask, 
    but defaults to 2mm resolution, and returns the 2mm (91, 109, 91)
    version of the mask, NOT the (99, 117, 95) version.

    Load the MNI152 whole-b_pos.nirain mask.

    This function takes the whole-brain MNI152 T1 template and threshold it,
    in order to obtain the corresponding whole-brain mask.

    .. versionadded:: 0.2.5

    Parameters
    ----------
    resolution: int, default=1
        If resolution is different from 1, the template loaded is first
        re-sampled with the specified resolution.

        .. versionadded:: 0.8.1

    threshold : float, default=0.2
        Values of the MNI152 T1 template above this threshold will be included.

    Returns
    -------
    mask_img : Nifti1Image, image corresponding to the whole-brain mask.

    Notes
    -----
    Refer to load_mni152_template function for more information about the
    MNI152 T1 template.

    See Also
    --------
    nilearn.datasets.load_mni152_template : for details about version of the
        MNI152 T1 template and related.

    """
    resolution = resolution or 2

    # Load MNI template
    target_img = load_mni152_template(resolution=resolution)
    mask_voxels = (nli.get_data(target_img) > threshold).astype("int8")
    mask_img = nli.new_img_like(target_img, mask_voxels)

    if resolution == 2:
        data = np.zeros((91, 109, 91))
        affine = np.array([[  -2.,    0.,    0.,   90.],
       [   0.,    2.,    0., -126.],
       [   0.,    0.,    2.,  -72.],
       [   0.,    0.,    0.,    1.]])
        target = nib.Nifti1Image(data, affine)
        mask_img = nli.resample_to_img(mask_img, target, interpolation="nearest")

    return mask_img