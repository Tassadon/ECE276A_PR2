
import numpy as np
from utils import read_canonical_model, load_pc, visualize_icp_result
from sklearn.neighbors import KDTree

if __name__ == "__main__":
  obj_name = 'drill' # drill or liq_container
  num_pc = 4 # number of point clouds

  source_pc = read_canonical_model(obj_name)
  
  print(source_pc.shape)
  #print(source_pc[0])
  for i in range(num_pc):

    target_pc = load_pc(obj_name, i) #target point cloud
    tree = KDTree(target_pc)

    m = target_pc
    z = source_pc

    R = np.eye(3)
    R_total = R
    p = np.array([np.mean(target_pc,axis=0)-np.mean(source_pc,axis=0)])
    p_total = p
    print("new epoch")

    for i in range(20):

      z = z @ R.T + p #nx3
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
    
    z = z @ R.T + p

    z_1 = source_pc
    m_1 = z

    z_1average = np.mean(z_1,axis=0)
    m_1average = np.mean(m_1,axis=0)

    z_1bar = z_1 - z_1average
    m_1bar = m_1 - m_1average
    Q_fun = m_1bar.T @ z_1bar

    U, S, Vh = np.linalg.svd(Q_fun)

    R_final = U @ np.diag([1,1,np.linalg.det(U @ Vh)]) @ Vh
    p_final = m_1average - R_final @ z_1average

    # estimated_pose, you need to estimate the pose with ICP
    pose = np.eye(4)
    pose[:3,:3] = R_final
    pose[:-1,-1] = p_final
    print(pose)
    # visualize the estimated result

    visualize_icp_result(source_pc, target_pc, pose)
    