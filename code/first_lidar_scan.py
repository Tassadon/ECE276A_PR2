import numpy as np
import math
import matplotlib.pyplot as plt
from pr2_utils import bresenham2D

def clean_lidar_data(lidar,index):

    indices = np.logical_and((lidar["ranges"][:,index] < lidar["range_max"]),(lidar["ranges"][:,index] > lidar["range_min"]))
    angles = np.linspace(lidar["angle_min"],lidar["angle_max"],1081)

    valid_ranges = lidar["ranges"][indices,index]
    valid_angles = angles[indices]
    
    coords = np.array([valid_ranges*np.cos(valid_angles),
                       valid_ranges*np.sin(valid_angles),
                       .51435*np.ones(len(valid_angles))])
    
    return coords.T

def load_lidar(path="../data",dataset=20):
    
    with np.load("../data/Hokuyo%d.npz"%dataset) as data:
      lidarlidar_angle_min = data["angle_min"] # start angle of the scan [rad]
      lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
      lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
      lidar_range_min = data["range_min"] # minimum range value [m]
      lidar_range_max = data["range_max"] # maximum range value [m]
      lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
      lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
      lidar = dict(data)

    return lidar

def cartesian_to_map_index(cartesian_x, cartesian_y):
    index_x = np.round((cartesian_x - 15)/.05)
    index_y = np.round((cartesian_y - -15)/.05)

    return index_x, index_y

if __name__ == '__main__':
    dataset = 20

    lidar_map = np.zeros((int(np.ceil((30 - -15) / .05 + 1)),
                      int(np.ceil((30 - -15) / .05 + 1))),dtype=np.int8) #DATA TYPE: char or int8
    x = np.load(f"./output/{dataset}_odometry_trajectory_to_lidar.npy")
    lidar = load_lidar(dataset=dataset)

    coords = clean_lidar_data(lidar,0)

    local_body_x = coords[:,0] + .13673 + x[0,0]
    local_body_y = coords[:,1] + x[0,1]
    global_x = local_body_x
    global_y = local_body_y

    lidar_x_index, lidar_y_index = cartesian_to_map_index(global_x, global_y)
    global_bot_x, global_bot_y = cartesian_to_map_index(x[0,0], x[0,1])
    print(lidar_x_index.shape)
    print("dingus:", lidar_x_index.shape[0])
    for row in range(lidar_x_index.shape[0]):
        free_cells = bresenham2D(global_bot_x, global_bot_y, lidar_x_index[row], lidar_y_index[row])
        free_cells = free_cells.astype(int)

        lidar_map[np.array(free_cells[0]), np.array(free_cells[1])] = 1


    first_map = (lidar_map*255).astype(int)

    plt.figure()
    plt.imshow(first_map)
    save_path = f"./output/{dataset}_first_lidar_map.png"
    #plt.savefig(save_path)
    plt.show()
    plt.pause(5)