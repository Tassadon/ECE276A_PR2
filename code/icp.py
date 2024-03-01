import numpy as np
from sklearn.neighbors import KDTree

def icp_transform(source,target,iterations=20,p_init=None,R_init=None,tol=1e-6):

  tree = KDTree(target)
    
  m = target
  z = source

  if R_init is not None:
    R = R_init
  else:
    R = np.eye(3)

  if p_init is not None:
    p = p_init
  else:
    p = np.array([np.mean(target,axis=0)-np.mean(source,axis=0)])

  prev_error = 10e10
  #print("FUNKO")
  for i in range(iterations):
    z = z @ R.T + p #nx3
    dist, indices = tree.query(z,k=1)
    m = target[np.squeeze(indices)]
    #print("The MSE of this iteration is:",  np.average(dist**2))
    cur_error = np.average(dist**2)
    #print("error: ", cur_error)

    #print(cur_error)

    if np.abs(cur_error - prev_error) < tol:
      break
    else:
      prev_error = cur_error
    
    z_average = np.mean(z,axis=0)
    m_average = np.mean(m,axis=0)
    z_bar = z - z_average
    m_bar = m - m_average
    Q_matrix = m_bar.T @ z_bar #3x3 matrix
    U, _, Vh = np.linalg.svd(Q_matrix)
    middle = np.diag([1,1, np.linalg.det(U @ Vh) ])
    R = U @ middle @ Vh
    p = m_average - R @ z_average

  z = z @ R.T + p

  z_1 = source
  m_1 = z

  z_1average = np.mean(z_1,axis=0)
  m_1average = np.mean(m_1,axis=0)

  z_1bar = z_1 - z_1average
  m_1bar = m_1 - m_1average
  Q_fun = m_1bar.T @ z_1bar

  U, _, Vh = np.linalg.svd(Q_fun)

  R_final = U @ np.diag([1,1,np.linalg.det(U @ Vh)]) @ Vh
  p_final = m_1average - R_final @ z_1average

  # estimated_pose, you need to estimate the pose with ICP
  pose = np.eye(4)
  pose[:3,:3] = R_final
  pose[:-1,-1] = p_final

  return pose