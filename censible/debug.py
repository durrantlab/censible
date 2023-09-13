import itertools
import os
import molgrid

# The order of the grids:
gridOrder = [
    l + "_RECEPTOR" for l in list(molgrid.defaultGninaReceptorTyper.get_type_names())
]
gridOrder += [
    l + "_LIGAND" for l in list(molgrid.defaultGninaLigandTyper.get_type_names())
]

def grid_channel_to_xyz_file(grid_channel) -> str:
    """Converts a grid channel to a .xyz file for visualization in VMD.
    
    Args:
        grid_channel: A 3D numpy array representing a grid channel.
        
    Returns:
        A string representing the .xyz file.
    """

    threshold = 0.85

    x_max, y_max, z_max = grid_channel.shape
    pts = []
    for x, y in itertools.product(range(x_max), range(y_max)):
        for z in range(z_max):
            val = grid_channel[x, y, z]
            if val > threshold:
                pts.append(f"X {str(x)} {str(y)} {str(z)}")

    contents = str(len(pts)) + "\n"
    contents += "\n"
    contents += "\n".join(pts)
    return contents


def save_all_channels(input_batch_voxel):
    # TODO: Debug below
    for channel in range(len(input_batch_voxel[0])):
        name = gridOrder[channel]
        filename = name + ".tmp" + str(channel) + ".xyz"
        if os.path.exists(filename):
            os.remove(filename)
        summed = input_batch_voxel[0][channel].sum().item()
        if summed != 0:
            print(f"{name}\t{summed}")
            xyz = grid_channel_to_xyz_file(input_batch_voxel[0][channel])
            with open(filename, "w") as f:
                f.write(xyz)
