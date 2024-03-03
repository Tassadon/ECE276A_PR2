import numpy as np
import math
import matplotlib.pyplot as plt
from pr2_utils import bresenham2D
from tqdm import tqdm
import pickle

def clean_lidar_data(lidar,index):

    indices = np.logical_and((lidar["ranges"][:,index] < lidar["range_max"]),(lidar["ranges"][:,index] > lidar["range_min"]))
    angles = np.linspace(lidar["angle_min"],lidar["angle_max"],1081)

    valid_ranges = lidar["ranges"][indices,index]
    valid_angles = angles[indices]
    
    coords = np.array([valid_ranges*np.cos(valid_angles),
                       valid_ranges*np.sin(valid_angles)])
    
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
    index_x = np.round((cartesian_x - -15)/.05) #was -30 was -45
    index_y = np.round((cartesian_y - -15)/.05)

    return index_x, index_y

if __name__ == '__main__':
    dataset = 20
    p = np.load(f"./output/{dataset}_odometry_trajectory_to_lidar.npy")
    x = np.load(f"./output/{dataset}_scan_matching.npy")
    lidar_map = np.zeros((int(np.ceil((30 - -30) / .05 + 1)),
                      int(np.ceil((30 - -30) / .05 + 1))),dtype=np.int8) #DATA TYPE: char or int8
    print(x.shape)
    print(p.shape)
    #plt.plot(p[:,0],x[:,1])
    #plt.plot(x[:,0],x[:,1])
    lidar = load_lidar(dataset=dataset)

    lidar_map_ownership = {}

    for t in tqdm(range(np.shape(lidar["time_stamps"])[0])):
      
      coords = clean_lidar_data(lidar,t)
      w = x[t,2]
      R = np.array([[np.cos(w),-np.sin(w)],
                    [np.sin(w),np.cos(w)]])
      rot_coords = (R @ coords.T).T
      lidar_x = rot_coords[:,0] + x[t,0] + .13673
      lidar_y = rot_coords[:,1] + x[t,1]
      lidar_x_index, lidar_y_index = cartesian_to_map_index(lidar_x, lidar_y)
      global_bot_x, global_bot_y = cartesian_to_map_index(x[t,0], x[t,1])
      a = set()
      for row in range(lidar_x_index.shape[0]):
          try:
            free_cells = bresenham2D(global_bot_x, global_bot_y, lidar_x_index[row], lidar_y_index[row])
            free_cells = free_cells.astype(int)
            lidar_map[np.array(free_cells[0]), np.array(free_cells[1])] = 1

            a = a.union(set(map(lambda co: tuple(co),free_cells.T.tolist())))
            
          except IndexError:
            continue
      lidar_map_ownership[t] = a

    
      
    with open(f'./output/{dataset}_lidar_map_ownership.pkl','wb') as f:
       pickle.dump(lidar_map_ownership,f)

    first_map = (lidar_map*255).astype(int)

    plt.figure()
    plt.imshow(first_map)
    plt.title(f"occupancy map for dataset {dataset}")
    plt.savefig(f"./output/{dataset}_occupancy_mapping_scan_matching.png")
    plt.show()
    plt.pause(10)