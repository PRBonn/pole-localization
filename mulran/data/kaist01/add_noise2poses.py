import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
rand = np.random.randn
rand_normal = np.random.normal
np.random.seed(0)


def test_rand():
  a = np.zeros(10000)
  for idx in range(10000):
    a[idx] = 10 * rand_normal(scale=1)
  plt.hist(a)
  plt.show()


def add_noise(poses, noises_raw=np.array([0.01, 0.1*np.pi/180.0]), noise_scale=1):
  new_poses = []
  noises = noises_raw * noise_scale
  for idx in range(len(poses)):
    new_pose = np.identity(4)
    new_pose[0, 3] = poses[idx, 0, 3] + noises[0]*rand(1)
    new_pose[1, 3] = poses[idx, 1, 3] + noises[0]*rand(1)

    roll, pitch, yaw = euler_angles_from_rotation_matrix(poses[idx, :3, :3])
    yaw_noisy = yaw + noises[1] * rand(1)
    new_pose[:3, :3] = R.from_euler('xyz', np.array((roll, pitch, yaw_noisy)).squeeze()).as_matrix()
    new_poses.append(new_pose)

  return np.array(new_poses)


if __name__ == '__main__':

  pose_file = 'localization_poses01_seq02asmap.txt'
  calib_file = 'calib.txt'
  poses = load_poses(pose_file)
  print(poses.shape)

  relative_poses = []

  for idx in range(len(poses)):
    if idx == 0:
      relative_poses.append(poses[idx])
    else:
      relative_poses.append(np.linalg.inv(poses[idx-1]).dot(poses[idx]))

  new_relative_poses = add_noise(np.array(relative_poses))
  new_absolute_poses = add_noise(np.array(poses))

  # from new relative to new global poses
  new_poses = []
  for idx in tqdm(range(len(new_relative_poses))):
    new_pose = np.identity(4)
    if idx == 0:
      new_poses.append(new_relative_poses[idx])
    else:
      for j in range(idx+1):
        new_pose = new_pose.dot(new_relative_poses[j])
      new_poses.append(new_pose)
  new_poses = np.array(new_poses)

  plt.plot(poses[:, 0, 3], poses[:, 1, 3], 'r', label='ground truth 02')
  plt.plot(new_poses[:, 0, 3], new_poses[:, 1, 3], 'b', label='noisy poses')
  plt.show()

  save_kitti_poses('noisy_poses_1_0.01_0.1.txt', new_poses)
  save_kitti_poses('noisy_relative_poses_1_0.01_0.1.txt', new_relative_poses)
