"""This module provides utility functions for accessing data files."""

import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

def data_file_path(filename: str) -> str:
    """Get the path to a file in the data directory.
    
    Args:
        filename (str): The name of the file to get the path to.
        
    Returns:
        The path to the file in the data directory.
    """
    return script_dir + os.sep + filename