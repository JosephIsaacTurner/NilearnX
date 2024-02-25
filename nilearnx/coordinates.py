import nibabel as nib
import numpy as np

class Coordinate2mm:
    def __init__(self, coord=None, coord_space='voxel'):
        """
        Initialize the Coordinate2mm object with a coordinate and coordinate space.
        
        Parameters:
        - coord: A 3-element sequence representing the coordinate. Default is (0, 0, 0) if None is provided.
        - coord_space: The space of the coordinate provided ('voxel' or 'mni'). Default is 'voxel'.
        """
        self.coord = self._validate_and_convert_coord(coord)
        self.coord_space = coord_space
        self.affine = np.array([
            [-2., 0., 0., 90.],
            [0., 2., 0., -126.],
            [0., 0., 2., -72.],
            [0., 0., 0., 1.]
        ])
    
    def _validate_and_convert_coord(self, coord):
        """Validate and convert the input coordinate to a tuple if necessary."""
        if coord is None:
            return (0, 0, 0)
        elif isinstance(coord, (list, np.ndarray)) and len(coord) == 3:
            return tuple(int(x) for x in coord)
        elif isinstance(coord, tuple) and len(coord) == 3:
            return tuple(int(x) for x in coord)
        else:
            raise ValueError("Coordinate must be a 3-element sequence.")
    
    @property
    def mni_space_coord(self):
        """Converts and returns the coordinate in MNI space."""
        if self.coord_space == 'mni':
            return self.coord
        else:
            return self.voxel_to_mni_space(self.coord)

    @property
    def voxel_space_coord(self):
        """Converts and returns the coordinate in voxel space."""
        if self.coord_space == 'voxel':
            return self.coord
        else:
            return self.mni_to_voxel_space(self.coord)
    
    # @property
    # def anatomical_name(self):
    #     """Returns the anatomical name at the peak coordinate"""
    #     index = self.voxel_space_coord
    #     atlas_cort_name = [Atlas(os.path.join(atlases_dir, "HarvardOxford-cort-maxprob-thr0-2mm.nii.gz")).name_at_index(index)]
    #     atlas_sub_name = [Atlas(os.path.join(atlases_dir, "HarvardOxford-sub-maxprob-thr0-2mm.nii.gz")).name_at_index(index)]
    #     results = [x for x in list(set(atlas_cort_name + atlas_sub_name)) if x != 'No matching region found']
    #     if len(results) == 0:
    #         return "No matching region found"
    #     elif len(results) == 1:
    #         return results[0]
    #     return results

    def mni_to_voxel_space(self, world_coord=None):
        """
        Converts MNI space coordinates to voxel space using the inverse of the affine matrix.
        
        Parameters:
        - world_coord: The coordinates in MNI space to be converted (3-element sequence).
        
        Returns:
        - voxel_coordinates: The equivalent coordinates in voxel space as a tuple.
        """
        if world_coord is None:
            world_coord = self.coord
        inverse_affine = np.linalg.inv(self.affine)
        voxel_coordinates = nib.affines.apply_affine(inverse_affine, world_coord)
        return tuple(np.rint(voxel_coordinates).astype(int))

    def voxel_to_mni_space(self, voxel_coord=None):
        """
        Converts voxel space coordinates to MNI space using the affine matrix.
        
        Parameters:
        - voxel_coord: The coordinates in voxel space to be converted (3-element sequence).
        
        Returns:
        - mni_coordinates: The equivalent coordinates in MNI space as a tuple.
        """
        if voxel_coord is None:
            voxel_coord = self.coord
        world_coordinates = nib.affines.apply_affine(self.affine, voxel_coord)
        return tuple(np.rint(world_coordinates).astype(int))
    