import numpy as np
from load_data import *
import matplotlib.pyplot as plt

def preprocess(encoder_counts, ang_vel):

    left_wheel = (encoder_counts[1] + encoder_counts[3])/2
    right_wheel = (encoder_counts[0] + encoder_counts[2])/2

    v_t = np.add(left_wheel,right_wheel)*.0022/2
    imu_data = ang_vel[2]

    return v_t, imu_data


def next_state(x_0,v_t,omega_t,tau_t):
    return np.array([x_0 + tau_t*np.array([
                    v_t*sinc(omega_t*tau_t/2)*np.cos(x_0[2] + omega_t*tau_t/2),
                    v_t*sinc(omega_t*tau_t/2)*np.sin(x_0[2] + omega_t*tau_t/2),
                    omega_t])])

def sinc(x):
    return np.sin(x)/(x)


def get_states(v_t, encoder_timestamps, imu_data, imu_stamps):

    x = np.array([[0,0,0]])

    for t,enc in enumerate(v_t[1:]):
        cur_imu = imu_data[np.argmin(np.abs(imu_stamps-encoder_timestamps[t+1]))]
        tau = encoder_timestamps[t+1] - encoder_timestamps[t]
        x = np.vstack([x,next_state(x[-1],enc/tau,cur_imu,tau)])

    return x

def drive_and_data():
    encoder_counts, encoder_timestamps = load_encoders(path="/Users/justin/Documents/Homework/ECE 276A/ECE 276A Project 2/ECE276A_PR2/data/")
    ang_vel, linear_acc, imu_stamps = load_imu(path="/Users/justin/Documents/Homework/ECE 276A/ECE 276A Project 2/ECE276A_PR2/data/")
    v_t, yaw_data_acc = preprocess(encoder_counts, ang_vel)

    states = get_states(v_t, encoder_timestamps, yaw_data_acc, imu_stamps)

    return states

def drive():
    dataset = 20
    encoder_counts, encoder_timestamps = load_encoders(path="../data",dataset=dataset)
    ang_vel, linear_acc, imu_stamps = load_imu(path="../data",dataset=dataset)
    v_t, yaw_data_acc = preprocess(encoder_counts, ang_vel)

    states = get_states(v_t, encoder_timestamps, yaw_data_acc, imu_stamps)

    plt.plot(states[:,0],states[:,1])
    plt.grid()
    plt.show()
    plt.title(f"odometry trajectory for dataset {dataset}")
    plt.savefig(f"./output/odometry trajectory{dataset}")
    np.save(f"./output/odometry_trajectory_{dataset}",states)
    plt.pause(10)
    
if __name__ == "__main__":
    drive()
    pass