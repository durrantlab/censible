import itertools
def grid_channel_to_xyz_file(grid_channel):
    """Converts a grid channel to a .xyz file for visualization in VMD.
    
    Args:
        grid_channel: A 3D numpy array representing a grid channel.
        
    Returns:
        A string representing the .xyz file.
    """

    x_max, y_max, z_max = grid_channel.shape
    pts = []
    for x, y in itertools.product(range(x_max), range(y_max)):
        for z in range(z_max):
            val = grid_channel[x, y, z]
            if val > 1.5:
                pts.append(f"X {str(x)} {str(y)} {str(z)}")
    print(len(pts))
    print("")
    print("\n".join(pts))
