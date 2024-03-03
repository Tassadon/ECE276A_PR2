import numpy as np
from load_data import *
from drive_model import *
import tqdm 
from icp import icp_transform
from transforms3d.euler import mat2euler

def clean_lidar_data(lidar,index):

    indices = np.logical_and((lidar["ranges"][:,index] < lidar["range_max"]),(lidar["ranges"][:,index] > lidar["range_min"]))
    angles = np.linspace(lidar["angle_min"],lidar["angle_max"],1081)

    valid_ranges = lidar["ranges"][indices,index]
    valid_angles = angles[indices]
    
    coords = np.array([valid_ranges*np.cos(valid_angles),
                       valid_ranges*np.sin(valid_angles),
                       .51435*np.ones(len(valid_angles))])
    
    return coords.T

def clean_all_data_and_save(lidar):
    
    for i in range(lidar["ranges"].shape[1]):
        pass

def scan_and_match(v_t, encoder_timestamps, imu_data, imu_stamps, lidar,dataset):

    x = np.array([[0,0,0]])
    cum_trans = np.eye(4)

    p = np.array([[0,0,0]])
    Transition_List = []
    Transition_List.append(cum_trans)
    for t in tqdm.tqdm(range(len(lidar["time_stamps"][1:]))):

        cur_imu = imu_data[np.argmin(np.abs(imu_stamps-lidar["time_stamps"][t+1]))]
        enc = v_t[np.argmin(np.abs(encoder_timestamps-lidar["time_stamps"][t+1]))]
        tau = lidar["time_stamps"][t+1] - lidar["time_stamps"][t]
        x = np.vstack([x, next_state(x[-1],enc/tau,cur_imu,tau) ])

        #process lidar data

        R_init = np.array([[np.cos(cur_imu*tau), -np.sin(cur_imu*tau), 0],
                           [np.sin(cur_imu*tau), np.cos(cur_imu*tau),0],
                           [0,0,1]])

        step_trans = np.eye(4)
        step_trans[:-1,:-1] = R_init
        step_trans[:-1,-1] = np.array([x[-1,0],x[-1,1],.51435])-np.array([p[-1,0],p[-1,1],.51435])

        source = clean_lidar_data(lidar,t)
        source = (cum_trans @ np.hstack([source,np.ones((source.shape[0],1))]).T).T
        
        source = source[:,:-1]

        target = clean_lidar_data(lidar,t+1)
        target = (step_trans @ cum_trans @ np.hstack([target,np.ones((target.shape[0],1))]).T).T
        target = target[:,:-1]

        Step_Transition = icp_transform(source, target,
                              R_init=R_init,
                              p_init=np.array([x[-1,0],x[-1,1],.51435])-np.array([p[-1,0],p[-1,1],.51435]),
                              iterations=20,tol=10e-10)
        
        cum_trans = Step_Transition @ cum_trans
        Transition_List.append(cum_trans)
        p = np.vstack([ p, np.array([cum_trans[0,-1], cum_trans[1,-1], x[-1,2]]) ])

    plt.plot(x[:,0],x[:,1])
    plt.plot(p[:,0],p[:,1])
    plt.show()
    plt.title("scan matching correction trajectory")
    plt.legend(["odometry trajectory","scan matching trajectory"])
    plt.pause(10)
    plt.savefig(f"{dataset}_scan_matching")
    np.save(f"./output/{dataset}_odometry_trajectory_to_lidar",x)
    np.save(f"./output/{dataset}_scan_matching",p)
    np.save(f"./output/{dataset}_poses",Transition_List)
    
    
    return p, x

if __name__ == "__main__":
    dataset = 21
    encoder_counts, encoder_timestamps = load_encoders(path="../data/",dataset=dataset)
    ang_vel, linear_acc, imu_stamps = load_imu(path="../data/",dataset=dataset)
    v_t, yaw_data_acc = preprocess(encoder_counts, ang_vel)

    lidar = load_lidar(dataset=dataset)
    print(list(lidar.keys()))
    #print(clean_lidar_data(lidar,0)[:,0])
    p,x = scan_and_match(v_t,encoder_timestamps,yaw_data_acc,imu_stamps, lidar,dataset)