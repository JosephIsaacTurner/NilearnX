# NilearnX
Extension of the popular nilearn library, with included extra functionality

### Installation

```bash
git clone https://github.com/josephisaacturner/nilearnx
cd nilearnx
pip install .
```

### Additional functionality beyond standard nilearn

```python
from nilearnx import extensions

path = 'my_nifti.nii' # path to nifti file
img = extensions.apply_mni_mask(path) # Returns the 3d image after masking with the MNI template
```
