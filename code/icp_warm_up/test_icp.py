
import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
from sklearn.neighbors import KDTree

if __name__ == "__main__":
  obj_name = 'drill' # drill or liq_container
  num_pc = 4 # number of point clouds

  source_pc = read_canonical_model(obj_name)
  
  print(source_pc.shape)
  print(source_pc[0])
  for i in range(num_pc):

    target_pc = load_pc(obj_name, i) #target point cloud
    tree = KDTree(target_pc)
    
    #m = target_pc
    #z = source_pc

    R = np.eye(3)
    p = np.array([np.mean(target_pc,axis=0)-np.mean(source_pc,axis=0)])
    print("new epoch")

    for i in range(20):

      z = source_pc @ R.T + p #nx3
      print(p)
      dist, indices = tree.query(z,k=1)

      m = target_pc[np.squeeze(indices)]

      print("The MSE of this iteration is:",  np.average(dist**2))
      
      z_average = np.mean(z,axis=0)

      m_average = np.mean(m,axis=0)


      z_bar = z - z_average
      m_bar = m - m_average

      Q_matrix = m_bar.T @ z_bar #3x3 matrix
      U, S, Vh = np.linalg.svd(Q_matrix)
      
      middle = np.diag([1,1, np.linalg.det(U @ Vh) ])

      R = U @ middle @ Vh

      p = m_average - R @ z_average
    
    # estimated_pose, you need to estimate the pose with ICP
    
    pose = np.eye(4)
    pose[:3,:3] = R
    pose[:-1,-1] = p
    print(pose)
    # visualize the estimated result

    visualize_icp_result(source_pc, target_pc, pose)
    break
