import os
from nilearn import image
import pandas as pd
from . import extensions
from nibabel import Nifti1Image
import numpy as np

def load_images(df, col_name_to_path):
    """Load images from file paths in a DataFrame.

    This function loads images from file paths stored in a DataFrame
    column and adds the loaded images as a new column.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the file paths.

    col_name_to_path : str
        Name of the DataFrame column containing the file paths.

    Returns:
    --------
    df : pandas DataFrame
        DataFrame with the loaded images added as a new column.
    """

    df = validate_dataframe(df)
    
    # Extract the short file name without extension from col_name_to_path
    target_col_name = find_suitable_col_name(df, 
                                             os.path.splitext(os.path.basename(col_name_to_path))[0] + "_img")

    # Apply image.load_img() to each path in the specified column
    df[target_col_name] = df[col_name_to_path].apply(image.load_img)
    # Iterate through dataframe and load images individually for debugging of loading errors
    # for index, row in df.iterrows():
    #     try:
    #         df.at[index, target_col_name] = image.load_img(row[col_name_to_path])
    #     except Exception as e:
    #         print(f"Error loading image: {e}")
    #         display(row)

    return df, target_col_name

def apply_mni_mask(df, img_col_name):
    """Apply MNI mask to Nifti images in a DataFrame.

    If the specified column doesn't contain Nifti images,
    attempt to load them using the load_images function.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the Nifti images.

    img_col_name : str
        Name of the column containing Nifti images.

    Returns:
    --------
    df : pandas DataFrame
        DataFrame with MNI mask applied to the Nifti images.
    """

    if not isinstance(df.iloc[0][img_col_name], Nifti1Image):
        print(f"{img_col_name} does not contain a Nifti object. Attempting to load...")
        df, new_col_name = load_images(df, img_col_name)
        img_col_name = new_col_name

    target_col_name = find_suitable_col_name(df, f"{img_col_name}_mni_masked")
    df[target_col_name] = df[img_col_name].apply(extensions.apply_mni_mask_fast)

    return df, target_col_name

def apply_custom_mask(df, img_col_name, masking_img_col_name, threshold=0.95):
    """Apply mask to Nifti images in a DataFrame.

    If the specified columns don't contain Nifti images,
    attempt to load them using the load_images function.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the Nifti images.

    img_col_name : str
        Name of the column containing Nifti images.

    masking_img_col_name : str
        Name of the column containing significance images.

    threshold : float, optional
        Significance level (default is 0.95).

    Returns:
    --------
    df : pandas DataFrame
        DataFrame with mask applied to the Nifti images.
    """

    if not isinstance(df.iloc[0][img_col_name], Nifti1Image):
        print(f"{img_col_name} does not contain a Nifti object. Attempting to load...")
        df, new_col_name = load_images(df, img_col_name)
        img_col_name = new_col_name

    if not isinstance(df.iloc[0][img_col_name], Nifti1Image):
        print(f"{masking_img_col_name} does not contain a Nifti object. Attempting to load...")
        df, new_col_name = load_images(df, masking_img_col_name)
        masking_img_col_name = new_col_name

    target_col_name = find_suitable_col_name(df, f"{img_col_name}_masked")
    df[target_col_name] = df.apply(lambda x: extensions.apply_custom_mask(x[img_col_name], x[masking_img_col_name], threshold), axis=1)

    return df, target_col_name

def add_peak_coord(df, img_col_name):
    """Add peak coordinates to a DataFrame from Nifti images.

    If the specified column doesn't contain Nifti images,
    attempt to load them using the load_images function.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the Nifti images.

    img_col_name : str
        Name of the column containing Nifti images.

    Returns:
    --------
    df : pandas DataFrame
        DataFrame with peak coordinates added as a new column.

    target_col_name : str
        Name of the new column containing peak coordinates.
    """

    if not isinstance(df.iloc[0][img_col_name], Nifti1Image):
        print(f"{img_col_name} does not contain a Nifti object. Attempting to load...")
        df, new_col_name = load_images(df, img_col_name)
        img_col_name = new_col_name

    target_col_name = find_suitable_col_name(df, 'peak_coord')
    df[target_col_name] = df[img_col_name].apply(lambda x: extensions.return_peak_coordinate(x))

    anatomical_label_col_name = find_suitable_col_name(df, f"{target_col_name}_anatomical_label")

    df[anatomical_label_col_name] = df[target_col_name].apply(lambda x: extensions.anatomical_label_at_idx(x))

    return df, target_col_name

def find_local_maxima(df, img_col_name):
    """
    Enhances a DataFrame by finding local maxima in images specified in a column.
    It processes these images, directly modifying the DataFrame to append new columns
    with this information.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the images or paths to images.
    img_col_name : str
        Name of the column containing the images or paths.

    Returns:
    --------
    df : pandas DataFrame
        DataFrame enhanced with additional columns for local maxima and anatomical labels.
    """

    def process_image_row(row_index, img):
        """
        Process an individual image row, updating the DataFrame directly.
        """
        # Assume extensions.find_local_maxima(img) returns a DataFrame with local maxima
        results_df = extensions.find_local_maxima(img)

        # Make sure results_df has no missing data whatsoever.
        if results_df is None or len(results_df) == 0 or 'coord' not in results_df.columns or results_df.isnull().values.any() or results_df.isna().values.any():
            return 
        
        # Get anatomical labels for each maximum
        try: 
            results_df['anatomical_label'] = results_df['coord'].apply(lambda x: extensions.anatomical_label_at_idx(x))
            results_df['anatomical_label'] = results_df['anatomical_label'].apply(lambda x: x[-1] if isinstance(x, list) else x)
        except Exception as e:
            print(f"Error getting anatomical labels: {e}")
            display(results_df)
            return

        # Prepare data for DataFrame integration
        for idx, row in results_df.iterrows():
            for col_suffix in ['coord', 'value', 'anatomical_label']:
                col_name = f"max{idx+1}_{col_suffix}"
                if col_name not in df.columns:
                    df[col_name] = None
                if row_index in df.index:
                    df.at[row_index, col_name] = row[col_suffix]

    df = validate_dataframe(df)
    df = df.copy()
    # # Process each row and update the DataFrame directly
    for index, row in df.iterrows():
        process_image_row(index, row[img_col_name])

    return df


def to_filename(df, img_col_name, output_dir=None, filepath_col_name=None):
    """Save Nifti images from a DataFrame to files named according to another column.

    Assumes the column specified by filepath_col_name already exists and contains the desired filenames.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the Nifti images or paths to images.

    img_col_name : str
        Name of the column containing Nifti images or paths.

    output_dir : str, optional
        Output directory for the saved images (default is current working directory).

    filepath_col_name : str
        Column name from which to retrieve the filenames for the saved images.

    Returns:
    --------
    df : pandas DataFrame
        Original DataFrame with the Nifti images saved to file using filenames specified in another column.
    """
    if not isinstance(df.iloc[0][img_col_name], Nifti1Image):
        print(f"{img_col_name} does not contain a Nifti object. Attempting to load...")
        # Assuming load_images is a function defined elsewhere that updates df to include Nifti1Image objects
        df, img_col_name = load_images(df, img_col_name)

    if output_dir is None:
        output_dir = os.getcwd()
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Ensure filepath_col_name exists in the DataFrame
    if filepath_col_name not in df.columns:
        raise ValueError(f"The specified filepath_col_name '{filepath_col_name}' does not exist in the DataFrame.")

    for index, row in df.iterrows():
        img = row[img_col_name]
        filename = row[filepath_col_name]  # Retrieve the filename from the specified column
        full_path = os.path.join(output_dir, filename)
        try:
            img.to_filename(full_path)
        except Exception as e:
            print(f"Error saving image to {full_path}: {e}")

    return df

def validate_dataframe(df):
    """Validate if input is a pandas DataFrame or a CSV file path.

    If input is a CSV file path, attempt to read it into a DataFrame.

    Parameters:
    -----------
    df : pandas DataFrame or str
        Input to be validated.

    Returns:
    --------
    df : pandas DataFrame or None
        Validated DataFrame, or None if validation fails.
    """

    # If input is a string, attempt to read it as a CSV file
    if isinstance(df, str):
        try:
            df = pd.read_csv(df)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

    # If input is not a DataFrame, return None
    if not isinstance(df, pd.DataFrame):
        print("Input is not a pandas DataFrame.")
        return None

    return df

def find_suitable_col_name(df, col_name, i=0):
    """Find a suitable column name that doesn't exist in the DataFrame."""
    if col_name not in df.columns:
        return col_name
    return find_suitable_col_name(df, f"{col_name}{str(i)}", i+1)