import numpy as np

def generate_trap_grid():
    grid_size = (30, 30)
    grid = np.zeros(grid_size, dtype=int)

    for i in range(4, 6):
        grid[i, 28] = 1
    for j in range(1, 29):
        # grid[0, j] = 1
        # grid[1, j] = 1
        # grid[2, j] = 1
        # grid[3, j] = 1
        # grid[4, j] = 1
        # grid[6, j] = 1
        # grid[7, j] = 1
        # grid[8, j] = 1
        # grid[9, j] = 1
        for k in range(1, 30):
            if k == 5: continue
            grid[k, j] = 1

    np.save("grid_files/grid_100.npy", grid)

def zachs_shit():
    data = np.load('grid_files/grid_0.npy')

    data [15, 15] = 1
    data [15, 14] = 1
    data [16, 15] = 1
    data [17, 15] = 1
    data [16, 14] = 1
    data [16, 13] = 1
    data [16, 12] = 1
    data [17, 14] = 1
    data [17, 13] = 1
    data [17, 12] = 1
    data [18, 15] = 1
    data [18, 14] = 0
    data [18, 13] = 0
    data [18, 12] = 0
    data [19, 14] = 1
    data [19, 13] = 1
    data [19, 12] = 1
    data [19, 11] = 1
    data [15, 18] = 1
    data [18, 11] = 0

    np.save('grid_files/grid_0.npy', data)

generate_trap_grid()
# zachs_shit()