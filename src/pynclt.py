import os
import open3d as o3
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import pyquaternion as pq
import transforms3d as t3
import util

T_w_o = np.identity(4)
T_w_o[:3, :3] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
T_o_w = util.invert_ht(T_w_o)
eulerdef = 'sxyz'

csvdelimiter = ','
datadir = 'nclt/data'
resultdir = 'nclt'
snapshotfile = 'snapshot.npz'
sessionfile = 'sessiondata_sata.npz'

sessions = [
    '2012-01-08',
    '2012-01-15',
    '2012-01-22',
    '2012-02-02',
    '2012-02-04',
    '2012-02-05',
    '2012-02-12',
    '2012-02-18',
    '2012-02-19',
    '2012-03-17',
    '2012-03-25',
    '2012-03-31',
    '2012-04-29',
    '2012-05-11',
    '2012-05-26',
    '2012-06-15',
    '2012-08-04',
    '2012-08-20',
    '2012-09-28',
    '2012-10-28',
    '2012-11-04',
    '2012-11-16',
    '2012-11-17',
    '2012-12-01',
    '2013-01-10',
    '2013-02-23',
    '2013-04-05'
    ]

lat0 = np.radians(42.293227)
lon0 = np.radians(-83.709657)
re = 6378135.0
rp = 6356750
rns = (re * rp)**2.0 \
    / ((re * np.cos(lat0))**2.0 + (rp * np.sin(lat0))**2.0)**1.5
rew = re**2.0 / np.sqrt((re * np.cos(lat0))**2.0 + (rp * np.sin(lat0))**2.0)

veloheadertype = np.dtype({
    'magic': ('<u8', 0),
    'count': ('<u4', 8),
    'utime': ('<u8', 12),
    'pad': ('V4', 20)})
veloheadersize = 24

velodatatype = np.dtype({
    'x': ('<u2', 0),
    'y': ('<u2', 2),
    'z': ('<u2', 4),
    'i': ('u1', 6),
    'l': ('u1', 7)})
velodatasize = 8


def load_snapshot(sessionname):
    cloud = o3.PointCloud()
    trajectory = o3.LineSet()
    with np.load(os.path.join(resultdir, sessionname, snapshotfile)) as data:
        cloud.points = o3.Vector3dVector(data['points'])
        cloud.colors = o3.Vector3dVector(
            util.intensity2color(data['intensities'] / 255.0))
        
        trajectory.points = o3.Vector3dVector(data['trajectory'])
        lines = np.reshape(range(data['trajectory'].shape[0] - 1), [-1, 1]) \
                + [0, 1]
        trajectory.lines = o3.Vector2iVector(lines)
        trajectory.colors = o3.Vector3dVector(
            np.tile([0.0, 0.5, 0.0], [lines.shape[0], 1]))
    return cloud, trajectory


def view_snapshot(sessionname):
    cloud, trajectory = load_snapshot(sessionname)
    o3.draw_geometries([cloud, trajectory])


def pose2ht(pose):
    r, p, y = pose[3:]
    return t3.affines.compose(
        pose[:3], t3.euler.euler2mat(r, p, y, eulerdef), np.ones(3))


def latlon2xy(latlon):
    lat = latlon[:, [0]]
    lon = latlon[:, [1]]
    return np.hstack([np.sin(lat - lat0) * rns,
        np.sin(lon - lon0) * rew * np.cos(lat0)])


def data2xyzi(data, flip=True):
    xyzil = data.view(velodatatype)
    xyz = np.hstack(
        [xyzil[axis].reshape([-1, 1]) for axis in ['x', 'y', 'z']])
    xyz = xyz * 0.005 - 100.0

    if flip:
        R = np.eye(3)
        R[2, 2] = -1
        # xyz = xyz @ R
        xyz = np.matmul(xyz, R)

    return xyz, xyzil['i']


def save_trajectories():
    trajectorydir = os.path.join(resultdir, 'trajectories_gt')
    util.makedirs(trajectorydir)
    
    trajectories = [session(s).T_w_r_gt[::20, :2, 3] for s in sessions]
    for i in range(len(trajectories)):
        plt.clf()
        [plt.plot(t[:, 0], t[:, 1], color=(0.5, 0.5, 0.5)) \
            for t in trajectories]
        plt.plot(trajectories[i][:, 0], trajectories[i][:, 1], color='y')
        plt.savefig(os.path.join(trajectorydir, sessions[i] + '.svg'))


class session:
    def __init__(self, session):
        self.session = session
        self.dir = os.path.join(resultdir, self.session)

        try:
            data = np.load(os.path.join(self.dir, sessionfile))
            self.velofiles = data['velofiles']
            self.t_velo = data['t_velo']
            self.velorawfile = data['velorawfile']
            self.t_rawvelo = data['t_rawvelo']
            self.i_rawvelo = data['i_rawvelo']
            self.t_gt = data['t_gt']
            self.T_w_r_gt = data['T_w_r_gt']
            self.T_w_r_gt_velo = data['T_w_r_gt_velo']
            self.t_cov_gt = data['t_cov_gt']
            self.cov_gt = data['cov_gt']
            self.t_odo = data['t_odo']
            self.T_w_r_odo = data['T_w_r_odo']
            self.T_w_r_odo_velo = data['T_w_r_odo_velo']
            self.t_relodo = data['t_relodo']
            self.relodo = data['relodo']
            self.relodocov = data['relodocov']
            self.t_gps = data['t_gps']
            self.gps = data['gps']
        except:
            velodir = os.path.join(datadir, 'velodyne_data', 
                self.session + '_vel', 'velodyne_sync')
            #velodir = os.path.join(datadir, 'velodyne_data', 
            #    self.session + '_vel', 'velodyne_sync_part')
            self.velofiles = [os.path.join(velodir, file) \
                for file in os.listdir(velodir) \
                if os.path.splitext(file)[1] == '.bin']
            self.velofiles.sort()
            self.t_velo = np.array([
                int(os.path.splitext(os.path.basename(velofile))[0]) \
                    for velofile in self.velofiles])

            self.velorawfile = os.path.join(datadir, 'velodyne_data', 
                self.session + '_vel', 'velodyne_hits.bin')
            self.t_rawvelo = []
            self.i_rawvelo = []
            with open(self.velorawfile, 'rb') as file:
                data = np.array(file.read(veloheadersize))
                while data:
                    header = data.view(veloheadertype)
                    if header['magic'] != 0xad9cad9cad9cad9c:
                        break
                    self.t_rawvelo.append(header['utime'])
                    self.i_rawvelo.append(file.tell() - veloheadersize)
                    file.seek(header['count'] * velodatasize, os.SEEK_CUR)
                    data = np.array(file.read(veloheadersize))
            self.t_rawvelo = np.array(self.t_rawvelo)
            self.i_rawvelo = np.array(self.i_rawvelo)

            posefile = os.path.join(
                datadir, 'ground_truth', 'groundtruth_' + self.session + '.csv')
            posedata = np.genfromtxt(posefile, delimiter=csvdelimiter)
            posedata = posedata[np.logical_not(np.any(np.isnan(posedata), 1))]
            self.t_gt = posedata[:, 0]
            self.T_w_r_gt = np.stack([T_w_o.dot(pose2ht(pose_o_r)) \
                for pose_o_r in posedata[:, 1:]])
            self.T_w_r_gt_velo = np.stack(
                [self.get_T_w_r_gt(t) for t in self.t_velo])

            cov_gtfile = os.path.join(
                datadir, 'ground_truth_cov', 'cov_' + self.session + '.csv')
            cov_gt = np.genfromtxt(cov_gtfile, delimiter=csvdelimiter)
            self.t_cov_gt = cov_gt[:, 0]
            self.cov_gt = np.stack(
                [np.reshape(roc[[
                1,  2,  3,  4,  5,  6,
                2,  7,  8,  9, 10, 11,
                3,  8, 12, 13, 14, 15,
                4,  9, 13, 16, 17, 18,
                5, 10, 14, 17, 19, 20,
                6, 11, 15, 18, 20, 21
                ]], [6, 6]) for roc in cov_gt])

            sensordir = os.path.join(
                datadir, 'sensor_data', self.session + '_sen')
            odofile = os.path.join(sensordir, 'odometry_mu_100hz.csv')
            ododata = np.genfromtxt(odofile, delimiter=csvdelimiter)
            self.t_odo = ododata[:, 0]
            self.T_w_r_odo = np.stack([T_w_o.dot(pose2ht(pose_o_r)) \
                for pose_o_r in ododata[:, 1:]])
            self.T_w_r_odo_velo = np.stack(
                [self.get_T_w_r_odo(t) for t in self.t_velo])

            relodofile = os.path.join(sensordir, 'odometry_mu.csv')
            relodo = np.genfromtxt(relodofile, delimiter=csvdelimiter)
            self.t_relodo = relodo[:, 0]
            self.relodo = relodo[:, [1, 2, 6]]
            
            relodocovfile = os.path.join(sensordir, 'odometry_cov.csv')
            relodocov = np.genfromtxt(relodocovfile, delimiter=csvdelimiter)
            self.relodocov = np.stack(
                [np.reshape(roc[[
                1,  2,  3,  4,  5,  6,
                2,  7,  8,  9, 10, 11,
                3,  8, 12, 13, 14, 15,
                4,  9, 13, 16, 17, 18,
                5, 10, 14, 17, 19, 20,
                6, 11, 15, 18, 20, 21
                ]], [6, 6]) for roc in relodocov])

            gpsfile = os.path.join(sensordir, 'gps.csv')
            gps = np.genfromtxt(gpsfile, delimiter=csvdelimiter)[:, [0, 3, 4]]
            self.t_gps = gps[:, 0]
            self.gps = latlon2xy(gps[:, 1:])
            
            util.makedirs(self.dir)
            np.savez(os.path.join(self.dir, sessionfile), 
                velofiles=self.velofiles, 
                t_velo=self.t_velo, 
                velorawfile=self.velorawfile,
                t_rawvelo=self.t_rawvelo,
                i_rawvelo=self.i_rawvelo,
                t_gt=self.t_gt, 
                T_w_r_gt=self.T_w_r_gt,
                T_w_r_gt_velo=self.T_w_r_gt_velo,
                t_cov_gt=self.t_cov_gt,
                cov_gt=self.cov_gt,
                t_odo=self.t_odo,
                T_w_r_odo=self.T_w_r_odo,
                T_w_r_odo_velo=self.T_w_r_odo_velo,
                t_relodo=self.t_relodo,
                relodo=self.relodo,
                relodocov=self.relodocov,
                t_gps=self.t_gps,
                gps=self.gps)

    def get_velo(self, i):
        return data2xyzi(np.fromfile(self.velofiles[i]))

    def get_velo_raw(self, i):
        with open(self.velorawfile, 'rb') as file:
            data = np.array(file.read(veloheadersize))
            header = data.view(veloheadertype)
            data = np.fromfile(file, count=header['count']).view(velodatatype)
            xyz = np.empty([data.shape[0], 3])
            intensities = np.empty([data.shape[0], 1])
            for i in range(data.shape[0]):
                xyz[i], intensities[i] = data2xyzi(data[i])
        return xyz, intensities

    def get_T_w_r_gt(self, t):
        i = np.clip(np.searchsorted(self.t_gt, t), 1, self.t_gt.size - 1) \
            + np.array([-1, 0])
        return util.interpolate_ht(self.T_w_r_gt[i], self.t_gt[i], t)

    def get_T_w_r_odo(self, t):
        i = np.clip(np.searchsorted(self.t_odo, t), 1, self.t_odo.size - 1) \
            + np.array([-1, 0])
        return util.interpolate_ht(self.T_w_r_odo[i], self.t_odo[i], t)
        
    def save_snapshot(self):
        print(self.session)
        naccupoints = int(3e7)
        nscans = len(self.velofiles)
        nmaxpoints = naccupoints / nscans
        accupoints = np.full([naccupoints, 3], np.nan)
        accuintensities = np.empty([naccupoints, 1])

        ipstart = 0
        with progressbar.ProgressBar(max_value=nscans) as bar:
            for i in range(nscans):
                points, intensities = self.get_velo(i)
                npoints = min(points.shape[0], nmaxpoints)
                ip = np.random.choice(points.shape[0], npoints, replace=False)
                points = np.hstack([points[ip], np.ones([npoints, 1])]).T
                points = self.T_w_r_gt_velo[i].dot(points)[:3].T
                accupoints[ipstart:ipstart+npoints] = points
                intensities = intensities[ip].reshape([-1, 1])
                accuintensities[ipstart:ipstart+npoints] = intensities
                ipstart += npoints
                bar.update(i)
        trajectory = self.T_w_r_gt[:, :3, 3]

        util.makedirs(self.dir)
        np.savez(os.path.join(self.dir, snapshotfile),
            points=accupoints, intensities=accuintensities, 
            trajectory=trajectory)


if __name__ == '__main__':
    for s in sessions:
        session(s)
