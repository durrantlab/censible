def grid_channel_to_xyz_file(grid_channel):
    x_max, y_max, z_max = grid_channel.shape
    pts = []
    for x in range(x_max):
        for y in range(y_max):
            for z in range(z_max):
                val = grid_channel[x, y, z]
                if val > 1.5:
                    pts.append("X " + str(x) + " " + str(y) + " " + str(z))
    print(len(pts))
    print("")
    print("\n".join(pts))
