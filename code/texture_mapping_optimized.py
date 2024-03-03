import numpy as np
import math
import matplotlib.pyplot as plt
from pr2_utils import bresenham2D
from icp import icp_transform
from tqdm import tqdm
import cv2
import transforms3d

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

#best_trajectory = np.load(f"{dataset}_scan_matching.npy").T
x = np.load(f"./output/{dataset}_optimized_path.npy").T

external_camera_roll = 0.0 
external_camera_pitch = 0.36
external_camera_yaw = 0.021
external_camera_translation = np.array([0.18, 0.005, 0.36])

external_camera_matrix = np.eye(4)

external_camera_matrix[:3,:3] =  np.array([[ 0.9356905, -0.0196524,  0.3522742],
                                        [0.0209985,  0.9997795, -0.0000000],
                                        [-0.3521966,  0.0073972,  0.9358968 ]])
external_camera_matrix[:3, -1] = external_camera_translation

R_o_r = np.array([[0, -1, 0, 0],
                [0, 0, -1, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1]])

R_o_r = np.linalg.inv(R_o_r)

color_map = np.zeros((int(np.ceil((30 - -15) / .05 + 1)),
                      int(np.ceil((30 - -15) / .05 + 1)), 
                      3))

def cartesian_to_map_index(cartesian_x, cartesian_y, *args):
    index_x = np.round((cartesian_x - -15)/.05).astype(int) #was -30 was -45
    index_y = np.round((cartesian_y - -15)/.05).astype(int)

    return index_x, index_y

disp_closest_index = []
for rgb_stamp in rgb_stamps:
    disp_closest_index.append(np.argmin(abs(disp_stamps - rgb_stamp)))

trajectory_closest_index = []
for rgb_stamp in rgb_stamps:
    trajectory_closest_index.append(np.argmin(abs(encoder_stamps - rgb_stamp)))

x_closest = x[:, trajectory_closest_index]

for rgb_index in tqdm(range(len(rgb_stamps))):
    disparity_index = disp_closest_index[rgb_index]
    imd = cv2.imread(f"../../dataRGBD/Disparity{dataset}/disparity{dataset}_{disparity_index + 1}.png",cv2.IMREAD_UNCHANGED) # (480 x 640)
    imc = cv2.imread(f"../../dataRGBD/RGB{dataset}/rgb{dataset}_{rgb_index + 1}.png")[...,::-1] # (480 x 640 x 3)

    disparity = imd.astype(np.float32)

    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd

    v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]

    fx = 585.05108
    fy = 585.05108
    cx = 315.83800
    cy = 242.94140
    x = (u-cx) / fx * z
    y = (v-cy) / fy * z

    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0) / fx)
    rgbv = np.round((v * 526.37 + 16662.0) / fy)
    valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])

    rgbu_valid = rgbu[valid].astype(int)
    rgbv_valid = rgbv[valid].astype(int)

    points_optical_frame = np.array([x[valid], y[valid], z[valid], np.ones((len(x[valid])))])
    points_regular_frame = R_o_r@points_optical_frame 

    points_robot_frame = external_camera_matrix@points_regular_frame

    x_current = x_closest[0, rgb_index]
    y_current = x_closest[1, rgb_index]
    theta_current = x_closest[2, rgb_index]
    z_current = 0.147009
    global_translation_current = [x_current, y_current, z_current]

    global_transformation = np.eye(4)
    global_transformation[:3, :3] = transforms3d.euler.euler2mat(0, 0, theta_current)
    global_transformation[:3, -1] = global_translation_current

    points_global_frame = global_transformation@points_robot_frame

    global_x_indices = (points_global_frame[0,:] > -30) & (points_global_frame[0,:] < 30)
    global_y_indices = (points_global_frame[1,:] > -30) & (points_global_frame[1,:] < 30)
    global_xy_clip_ind = global_x_indices & global_y_indices
    
    floor_indices = (points_global_frame[2,:] > -30) & (points_global_frame[2,:] < 30)

    final_indices = global_xy_clip_ind & floor_indices #set logic

    floor_global_frame = points_global_frame[:, final_indices]

    floor_global_frame_x = floor_global_frame[0,:]
    floor_global_frame_y = floor_global_frame[1,:]

    floor_x_pixelated, floor_y_pixelated = cartesian_to_map_index(floor_global_frame_x, floor_global_frame_y)

    rgbu_final = rgbu_valid[final_indices]
    rgbv_final = rgbv_valid[final_indices]

    color_map[floor_x_pixelated, floor_y_pixelated] = imc[rgbv_final, rgbu_final]

color_map = color_map.astype(int)

plt.imshow(color_map)
plt.savefig(f"./output/{dataset}_texture_map_optimized.png")
plt.title(f"optimized texture map for dataset {dataset}")
plt.show()