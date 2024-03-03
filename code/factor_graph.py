import gtsam
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
import math
import gtsam.utils.plot as gtsam_plot
from sklearn.neighbors import KDTree
import pickle
from tqdm import tqdm

if __name__ == "__main__":
    dataset = 20
    odom = np.load(f"./output/{dataset}_odometry_trajectory_to_lidar.npy")
    odom_sm = np.load(f"./output/{dataset}_scan_matching.npy")
    poses = np.load(f"./output/{dataset}_poses.npy")
    
    with open(f'./output/{dataset}_lidar_map_ownership.pkl', 'rb') as f:
        occupancy_overlap = pickle.load(f)
    
    print(len(list(occupancy_overlap.keys())))
    
    graph = gtsam.NonlinearFactorGraph()
    prior_model = gtsam.noiseModel.Diagonal.Sigmas((.1, .1, .1))
    initial_estimate = gtsam.Values()

    odometry_model = gtsam.noiseModel.Diagonal.Sigmas((0.3, 0.3, 0.2))
    Between = gtsam.BetweenFactorPose2
    
    graph.add(gtsam.PriorFactorPose2(0, gtsam.Pose2(odom_sm[0,0], odom_sm[0,1], odom_sm[0,2]), prior_model))
    n = 30
    count = 0
    for t in tqdm(range(odom_sm.shape[0])):
        initial_estimate.insert(t,gtsam.Pose2(odom_sm[t,0], odom_sm[t,1], odom_sm[t,2]))
        if t != odom_sm.shape[0] - 1:
            count += 1

            graph.add(Between(t, t+1, gtsam.Pose2(odom_sm[t+1,0] - odom_sm[t,0],
                                                  odom_sm[t+1,1] - odom_sm[t,1],
                                                  mat2euler(poses[t+1] @ np.linalg.inv(poses[t]))[2] ),
                                                  odometry_model))
        '''
        if t!= 0 and t % n == 0:
            count += 1
            graph.add(Between(t-n, t, gtsam.Pose2(odom_sm[t,0] - odom_sm[t-n,0],
                                                  odom_sm[t,1] - odom_sm[t-n,1],
                                                  mat2euler(poses[t] @ np.linalg.inv(poses[t-n]))[2]), odometry_model))
                                                  '''
        
        for j in range(t,odom_sm.shape[0],n):
            if not occupancy_overlap[t].isdisjoint(occupancy_overlap[j]):
                count +=1
                graph.add(Between(t, j, gtsam.Pose2(odom_sm[j,0] - odom_sm[t,0],
                                                  odom_sm[j,1] - odom_sm[t,1],
                                                  mat2euler(poses[j] @ np.linalg.inv(poses[t]))[2]), 
                                                  odometry_model))
                                                  
            
    print("The number of constraints implemented:", count)

    # Optimize the initial values using a Gauss-Newton nonlinear optimizer
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)

    result = optimizer.optimize()
    p = []
    for i in range(odom_sm.shape[0]):
        p.append([result.atPose2(i).x(),result.atPose2(i).y(),result.atPose2(i).theta()])

    optimized = np.array(p)
    plt.plot(odom[:,0],odom[:,1])
    plt.plot(odom_sm[:,0],odom_sm[:,1])
    plt.plot(optimized[:,0],optimized[:,1])
    np.save(f"./output/{dataset}_optimized_path",optimized)
    plt.savefig(f"./output/{dataset}_optimized_trajectory")
    plt.title(f"Pose Graph Optimization on dataset {dataset}")
    plt.legend(["odometry trajectory","scan matching trajectory", "optimized trajectory"])
    plt.show()