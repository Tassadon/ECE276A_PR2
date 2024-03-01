import numpy as np
from load_data import *
from drive_model import *
import tqdm 
from icp import icp_transform

def clean_lidar_data(lidar,index):

    indices = np.logical_and((lidar["ranges"][:,index] < lidar["range_max"]),(lidar["ranges"][:,index] > lidar["range_min"]))
    angles = np.linspace(lidar["angle_min"],lidar["angle_max"],1081)

    valid_ranges = lidar["ranges"][indices,index]
    valid_angles = angles[indices]
    coords = np.array([valid_ranges*np.cos(valid_angles),
                       valid_ranges*np.sin(valid_angles),
                       .51435*np.ones(len(valid_angles))])
    
    return coords.T

def scan_and_match(v_t, encoder_timestamps, imu_data, imu_stamps, lidar):

    x = np.array([[0,0,0]])
    cum_trans = np.eye(4)

    p = np.array([[0,0,0]])

    for t in tqdm.tqdm(range(len(lidar["time_stamps"][1:]))):

        cur_imu = imu_data[np.argmin(np.abs(imu_stamps-lidar["time_stamps"][t+1]))]

        enc = v_t[np.argmin(np.abs(encoder_timestamps-lidar["time_stamps"][t+1]))]
        
        tau = lidar["time_stamps"][t+1] - lidar["time_stamps"][t]
        
        x = np.vstack([x, next_state(x[-1],enc/tau,cur_imu,tau) ])

        #process lidar data

        R_init = np.array([[np.cos(cur_imu*tau), -np.sin(cur_imu*tau), 0],
                           [np.sin(cur_imu*tau), np.cos(cur_imu*tau),0],
                           [0,0,1]])
        
        #R_init = np.eye(3)

        step_trans = np.eye(4)
        step_trans[:-1,:-1] = R_init
        step_trans[:-1,-1] = np.array([x[-1,0],x[-1,1],.51435])-np.array([x[-2,0],x[-2,1],.51435])
        print(step_trans)

        if t >=1:
            break

        source = clean_lidar_data(lidar,t)
        source = (cum_trans @ np.hstack([source,np.ones((source.shape[0],1))]).T).T
        
        source = source[:,:-1]

        target = clean_lidar_data(lidar,t+1)
        target = (cum_trans @ step_trans @ np.hstack([target,np.ones((target.shape[0],1))]).T).T
        target = target[:,:-1]

        #print("source", np.average(source,axis=0))
        #print("target", np.average(target,axis=0))

        Trans = icp_transform(target, source,
                              R_init=R_init,
                              p_init=np.array([x[-1,0],x[-1,1],.51435])-np.array([x[-2,0],x[-2,1],.51435]),
                              iterations=5)
        
        cum_trans = Trans @ cum_trans

        p = np.vstack([ p, cum_trans[:-1,-1] ])

    #print(p)
    
    plt.plot(x[:,0],x[:,1])
    plt.plot(p[:,0],p[:,1]) 
    plt.show()
    plt.pause(5)
    return p, x

if __name__ == "__main__":
    encoder_counts, encoder_timestamps = load_encoders(path="/Users/justin/Documents/Homework/ECE 276A/ECE 276A Project 2/ECE276A_PR2/data/")
    ang_vel, linear_acc, imu_stamps = load_imu(path="/Users/justin/Documents/Homework/ECE 276A/ECE 276A Project 2/ECE276A_PR2/data/")
    v_t, yaw_data_acc = preprocess(encoder_counts, ang_vel)

    lidar = load_lidar()
    print(list(lidar.keys()))
    #print(lidar["time_stamps"][0])

    #print(clean_lidar_data(lidar,0))
    p,x = scan_and_match(v_t,encoder_timestamps,yaw_data_acc,imu_stamps, lidar)
    