import numpy as np
import math
import matplotlib.pyplot as plt
from pr2_utils import bresenham2D

def load_encoders(path="../data",dataset=20):
  with np.load(path + "/Encoders%d.npz"%dataset) as data:
    #front right, front left, rear right, rear left
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

  return encoder_counts, encoder_stamps

def load_imu(path="../data",dataset=20):
    with np.load(path + "/Imu%d.npz"%dataset) as data:
      imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
      imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
      imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
    
    return imu_angular_velocity, imu_linear_acceleration, imu_stamps

if __name__ == '__main__':
  dataset = 20
  
  with np.load("../data/Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

  with np.load("../data/Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
    
  with np.load("../data/Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
  with np.load("../data/Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images



  lidar_theta_all = np.linspace(lidar_angle_min, lidar_angle_max, np.shape(lidar_ranges)[0])
  MAP = {}
  MAP['res']   = .05 #meters
  MAP['xmin']  =  -15  #meters
  MAP['ymin']  =  -15
  MAP['xmax']  =  30
  MAP['ymax']  =  30
  MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
  MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
  MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) #DATA TYPE: char or int8

  def cartesian_to_map_index(cartesian_x, cartesian_y, MAP):
      index_x = int((cartesian_x - MAP['xmin'])/MAP['res'])
      index_y = int((cartesian_y - MAP['ymin'])/MAP['res'])

      return index_x, index_y

  first_lidar_scan = lidar_ranges[:,0]
  all_grid_points = []
  for lidar_r, lidar_theta in zip(first_lidar_scan, lidar_theta_all):
      if lidar_r < lidar_range_min or lidar_r > lidar_range_max:
          continue
      local_sensor_x = lidar_r * np.cos(lidar_theta)
      local_sensor_y = lidar_r * np.sin(lidar_theta)

      local_body_x = local_sensor_x + .13673
      local_body_y = local_sensor_y
      global_x = local_body_x
      global_y = local_body_y

      lidar_x_index, lidar_y_index = cartesian_to_map_index(global_x, global_y, MAP)
      global_bot_x, global_bot_y = cartesian_to_map_index(0, 0, MAP)

      free_cells = bresenham2D(global_bot_x, global_bot_y, lidar_x_index, lidar_y_index)
      free_cells = free_cells.astype(int)

      MAP['map'][np.array(free_cells[0]), np.array(free_cells[1])] = 1

  first_map = (MAP["map"]*255).astype(int)

  plt.figure()
  plt.imshow(first_map, interpolation='nearest')
  save_path = str(20) + "_first_map.png"
  #plt.savefig(save_path)
  plt.show()
  plt.pause(10)