import numpy as np
from sklearn.neighbors import NearestNeighbors

def icp(source_pc,target_pc,iterations=20):

    model = NearestNeighbors(n_neighbors=1).fit(source_pc)

    R = np.random.rand(3,3)
    p = np.random.rand(1,3)

    print("new epoch")
    for i in range(20):
      #print(target_pc.shape)
      m = target_pc @ R + p
      
      _, indices = model.kneighbors(m)
      z = source_pc[np.squeeze(indices)]
      print(np.sum(np.linalg.norm(z - m,axis=1)**2)/m.shape[0])

      z_average = np.mean(z,axis=0)
      m_average = np.mean(m,axis=0)

      z_bar = z - z_average
      m_bar = m - m_average
      
      Q_matrix = m_bar.T @ z_bar
      U, S, Vh = np.linalg.svd(Q_matrix)
      
      middle = np.eye(3)
      middle[2,2] = np.linalg.det(U)*np.linalg.det(Vh)
      print(middle)
      R = U @ middle @ Vh
      p = m_average - R @ z_average

    pose = np.eye(4)
    pose[:3,:3] = R
    pose[:-1,-1] = p
    print(pose)

    return pose