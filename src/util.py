import os
import math
import numpy as np
import open3d as o3
import transforms3d as t3
import pyquaternion as pq


def invert_ht(ht):
    ht = np.tile(ht, [1, 1, 1])
    iht = np.tile(np.identity(4), [ht.shape[0], 1, 1])
    iht[..., :3, :3] = ht[..., :3, :3].transpose(0, 2, 1)
    iht[..., :3, [3]] = -np.matmul(iht[..., :3, :3], ht[..., :3, [3]])
    return iht.squeeze()


def create_wire_box(edgelengths, color=[0.0, 0.0, 0.0]):
    lineset = o3.LineSet()
    x, y, z = edgelengths
    lineset.points = o3.Vector3dVector([[0, 0, 0], [x, 0, 0], [0, y, 0], 
        [x, y, 0], [0, 0, z], [x, 0, z], [0, y, z], [x, y, z]])
    lineset.lines = o3.Vector2iVector([[0, 1], [1, 3], [3, 2], [2, 0],
        [0, 4], [1, 5], [3, 7], [2, 6],
        [4, 5], [5, 7], [7, 6], [6, 4]])
    lineset.colors = o3.Vector3dVector(np.tile(color, [len(lineset.lines), 1]))
    return lineset


def intensity2color(intensity):
    return np.tile(np.reshape(intensity * 0.8, [-1, 1]), [1, 3])


def xyp2ht(xyp):
    ht = np.tile(np.identity(4), [int(xyp.size / 3), 1, 1])
    cp = np.cos(xyp[..., 2])
    sp = np.sin(xyp[..., 2])
    ht[..., :2, 3] = xyp[..., :2]
    ht[..., 0, 0] = cp
    ht[..., 0, 1] = -sp
    ht[..., 1, 0] = sp
    ht[..., 1, 1] = cp
    return ht.squeeze()


def ht2xyp(ht):
    ht = np.tile(ht, [1, 1, 1])
    p = np.arctan2(ht[..., 1, 0], ht[..., 0, 0])
    return np.hstack([ht[..., :2, 3], np.reshape(p, [-1, 1])]).squeeze()


def interpolate_ht(ht, t, tq):
    amount = np.clip((tq - t[0]) / np.diff(t), 0.0, 1.0)
    pos = ht[0, :3, 3] + amount * np.diff(ht[:, :3, 3], axis=0).squeeze()
    q = [pq.Quaternion(matrix=m) for m in ht]
    qq = pq.Quaternion.slerp(q[0], q[1], amount=amount)
    return t3.affines.compose(pos, qq.rotation_matrix, np.ones(3))


def project_xy(ht):
    htp = np.identity(4)
    htp[:2, 0] = ht[:2, 0] / np.linalg.norm(ht[:2, 0])
    htp[:2, 1] = [-htp[1, 0], htp[0, 0]]
    htp[:2, 3] = ht[:2, 3]
    return htp


def xyzi2pc(xyz, intensities=None):
    pc = o3.PointCloud()
    pc.points = o3.Vector3dVector(xyz)
    if intensities is not None:
        pc.colors = o3.Vector3dVector(intensity2color(intensities / 255.0))
    return pc


def average_angles(angles, weights=None):
    if weights is None:
        weights = np.ones(angles.shape[0])
    x = np.cos(angles) * weights
    y = np.sin(angles) * weights
    return np.arctan2(np.sum(y), np.sum(x))


def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass

