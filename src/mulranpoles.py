import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import scipy.interpolate
import scipy.special
import cluster
import particlefilter
import util
from utils import *
import poles_extractor

np.random.seed(0)

mapinterval = 1.5
mapdistance = 1.5

n_mapdetections = 6
n_locdetections = 2
n_localmaps = 3

def get_globalmapname():
    return 'globalmap_{:.0f}_{:.2f}_{:.2f}'.format(
        n_mapdetections, mapinterval, 0.08)


def get_locfileprefix():
    return 'localization_{:.0f}_{:.2f}_{:.2f}_{:.0f}'.format(
        n_mapdetections, mapinterval, 0.08,1000)


def get_localmapfile():
    return 'localmaps_{:.0f}_{:.2f}_{:.2f}.npz'.format(
        n_mapdetections, mapinterval, 0.08)


def get_evalfile():
    return 'evaluation_{:.0f}_{:.2f}_{:.2f}.npz'.format(
        n_mapdetections, mapinterval, 0.08)

pose_file = 'mulran/data/kaist02/localization_poses02_seq02asmap.txt'
poses = load_poses(pose_file)

def get_map_indices(poses_):
    distance = np.hstack([0.0, np.cumsum(np.linalg.norm(
        np.diff(poses_[:, :3, 3], axis=0), axis=1))])
    istart = []
    imid = []
    iend = []
    i = 0
    j = 0
    k = 0
    for id, d in enumerate(distance):
        if d >= i * mapinterval:
            istart.append(id)
            i += 1
        if d >= j * mapinterval + 0.5 * mapdistance:
            imid.append(id)
            j += 1
        if d > k * mapinterval + mapdistance:
            iend.append(id)
            k += 1
    return istart[:len(iend)], imid[:len(iend)], iend

def save_global_map():
    scan_folder = 'mulran/data/kaist02/velodyne'
    scans = load_files(scan_folder)
    poleparams = np.empty([0, 3])
    istart, imid, iend = get_map_indices(poses)
    localmappos = poses[imid, :2, 3]
    imaps = range(localmappos.shape[0])
    with progressbar.ProgressBar(max_value=len(imaps)) as bar:
        for iimap, imap in enumerate(imaps):
            iscan = imid[imap]
            scan = load_vertex(scans[iscan])
            localpoleparam = poles_extractor.detect_poles(scan, neighbourthr = 0.3, min_point_num = 7, dis_thr = 0.15, width_thr = 3, fov_up=16.0, fov_down=-16.0, proj_H = 64, proj_W = 500, lowest=-1.2, highest=3, lowthr = 0.8, highthr = -0.7, totalthr = 0.5)
            
            localpoleparam_xy = localpoleparam[:, :2]
            localpoleparam_xy = localpoleparam_xy.T
            localpoleparam_xy = np.vstack([localpoleparam_xy, np.zeros_like(localpoleparam_xy[0]), np.ones_like(localpoleparam_xy[0])]) #4*n
            localpoleparam_xy = np.matmul(poses[imid[imap]], localpoleparam_xy)
            localpoleparam[:, :2] = localpoleparam_xy[:2,:].T

            poleparams = np.vstack([poleparams, localpoleparam])
            bar.update(iimap)

    xy = poleparams[:, :2]
    a = poleparams[:, [2]]
    boxes = np.hstack([xy - a, xy + a])
    clustermeans = np.empty([0, 3])
    for ci in cluster.cluster_boxes(boxes):
        ci = list(ci)
        if len(ci) < n_mapdetections:
            continue
        clustermeans = np.vstack([clustermeans, np.average(
            poleparams[ci, :], axis=0)])

    globalmapfile = os.path.join('mulran', get_globalmapname() + '.npz')
    np.savez(globalmapfile, 
        polemeans=clustermeans)
    plot_global_map(globalmapfile)

def plot_global_map(globalmapfile):
    data = np.load(globalmapfile)
    x, y = data['polemeans'][:, :2].T
    plt.clf()
    plt.axis('off')
    plt.scatter(x, y, s=1, c='k', marker='.')
    plt.savefig('test' + '.svg')

pose_file_test = 'mulran/data/kaist01/localization_poses01_seq02asmap.txt'
poses_test = load_poses(pose_file_test)

pose_file_odo = 'mulran/data/kaist01/noisy_poses_1_0.01_0.1.txt'
poses_odo = load_poses(pose_file_odo)

pose_relative_file = 'mulran/data/kaist01/noisy_relative_poses_1_0.01_0.1.txt'
poses_relative = load_poses(pose_relative_file)

def save_local_maps():
    scan_folder_test = 'mulran/data/kaist01/velodyne'
    scans_test = load_files(scan_folder_test)
    istart, imid, iend = get_map_indices(poses_test)
    maps = []
    with progressbar.ProgressBar(max_value=len(iend)) as bar:
        for i in range(len(iend)):
            iscan = imid[i]
            T_w_r = poses_odo[imid[i]]
            scan = load_vertex(scans_test[iscan])
            localpoleparam = poles_extractor.detect_poles(scan, neighbourthr = 0.3, min_point_num = 7, dis_thr = 0.15, width_thr = 3, fov_up=16.0, fov_down=-16.0, proj_H = 64, proj_W = 500, lowest=-1.2, highest=3, lowthr = 0.8, highthr = -0.7, totalthr = 0.5)
            
            map = {'poleparams': localpoleparam, 'T_w_r': T_w_r,
                'istart': istart[i], 'imid': imid[i], 'iend': iend[i]}
            maps.append(map)
            bar.update(i)

    np.savez(os.path.join('mulran', get_localmapfile()), maps=maps)

def localize():
    print("start localize")
    mapdata = np.load(os.path.join('mulran', get_globalmapname() + '.npz'))
    polemap = mapdata['polemeans'][:, :2]
    polevar = 1.00
    locdata = np.load(os.path.join('mulran', get_localmapfile()), allow_pickle=True)['maps']
    polepos_m = []
    polepos_w = []
    for i in range(len(locdata)):
        n = locdata[i]['poleparams'].shape[0]
        pad = np.hstack([np.zeros([n, 1]), np.ones([n, 1])])
        polepos_m.append(np.hstack([locdata[i]['poleparams'][:, :2], pad]).T)
        polepos_w.append(locdata[i]['T_w_r'].dot(polepos_m[i]))
    istart = 0
    T_w_r_start = poses_test[istart]
    filter = particlefilter.particlefilter(2000, 
        T_w_r_start, 3, np.radians(5.0), polemap, polevar, d_max=1.5)
    filter.estimatetype = 'best'
    filter.minneff = 0.5

    t_velo = range(poses_relative.shape[0])
    t_relodo = range(poses_relative.shape[0])

    imap = 0
    while imap < locdata.shape[0] - 1 and \
            t_velo[locdata[imap]['iend']] < t_relodo[istart]:
        imap += 1
    T_w_r_est = np.full([len(t_relodo), 4, 4], np.nan)
    T_w_r_est[istart] = filter.estimate_pose()
    istart += 1
    with progressbar.ProgressBar(max_value=len(t_relodo)-istart) as bar:
        for i in range(istart, len(t_relodo)):
            relodo = util.ht2xyp(
                util.invert_ht(poses_test[i-1]).dot(poses_test[i]))
            relodocov = np.diag((0.02 * relodo)**2)
            relodo = np.random.multivariate_normal(relodo, relodocov)
            filter.update_motion(relodo, relodocov)
            T_w_r_est[i] = filter.estimate_pose()
            t_now = t_relodo[i]
            if imap < locdata.shape[0]:
                t_end = t_velo[locdata[imap]['iend']]
                if t_now >= t_end:
                    imaps = range(imap, np.clip(imap-n_localmaps, -1, None), -1)
                    xy = np.hstack([polepos_w[j][:2] for j in imaps]).T
                    a = np.vstack([ld['poleparams'][:, [2]] \
                        for ld in locdata[imaps]])
                    boxes = np.hstack([xy - a, xy + a])
                    ipoles = set(range(polepos_w[imap].shape[1]))
                    iactive = set()
                    for ci in cluster.cluster_boxes(boxes):
                        if len(ci) >= n_locdetections:
                            iactive |= set(ipoles) & ci
                    iactive = list(iactive)
                    if iactive:
                        t_mid = t_velo[locdata[imap]['imid']]
                        T_w_r_mid = poses_odo[t_mid]
                        T_w_r_now = poses_odo[t_now]
                        T_r_now_r_mid = util.invert_ht(T_w_r_now).dot(T_w_r_mid)
                        polepos_r_now = T_r_now_r_mid.dot(
                            polepos_m[imap][:, iactive])
                        filter.update_measurement(polepos_r_now[:2].T)
                        T_w_r_est[i] = filter.estimate_pose()
                    imap += 1

            bar.update(i)
    filename = os.path.join('mulran', get_locfileprefix() \
        + datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.npz'))
    np.savez(filename, T_w_r_est=T_w_r_est)

def plot_trajectories():
    trajectorydir = os.path.join(
        'mulran', 'trajectories_est_{:.0f}_{:.0f}_{:.2f}'.format(
            n_mapdetections, n_locdetections, mapinterval))
    pgfdir = os.path.join(trajectorydir, 'pgf')
    util.makedirs(trajectorydir)
    util.makedirs(pgfdir)
    mapdata = np.load(os.path.join('mulran', get_globalmapname() + '.npz'))
    polemap = mapdata['polemeans']

    files = [file for file \
        in os.listdir('mulran') \
            if file.startswith(get_locfileprefix())]
    for file in files:
        T_w_r_est = np.load(os.path.join(
            'mulran', file))['T_w_r_est']
        plt.clf()
        plt.axis('off')
        plt.scatter(polemap[:, 0], polemap[:, 1], 
            s=1, c='k', marker='.')
        plt.plot(poses_test[:, 0, 3], 
            poses_test[:, 1, 3], color=(0.5, 0.5, 0.5))
        plt.plot(T_w_r_est[:, 0, 3], T_w_r_est[:, 1, 3], 'b')
        filename = "trajectories"+ datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')
        plt.savefig(os.path.join(trajectorydir, filename + '.svg'))
        plt.savefig(os.path.join(pgfdir, filename + '.pgf'))

def evaluation():
    pose_file = 'mulran/data/kaist01/localization_poses01_seq02asmap.txt'

    gt_poses = np.array(load_poses(pose_file))
    gt_xy_raw = gt_poses[:, :2, 3]
    gt_yaw = []
    for pose in gt_poses:
        gt_yaw.append(euler_angles_from_rotation_matrix(pose[:3, :3])[2])
    gt_yaw_raw = np.array(gt_yaw)

    est_pose_file = [file for file \
            in os.listdir('mulran') \
                if file.startswith(get_locfileprefix())]
    T_w_r_est = np.load('mulran/'+est_pose_file[0])['T_w_r_est']

    est_xy_raw = T_w_r_est[:, :2, 3]

    est_yaw = []
    for pose in T_w_r_est:
        est_yaw.append(euler_angles_from_rotation_matrix(pose[:3, :3])[2])
    est_yaw_raw = np.array(est_yaw)

    diffs_xy = np.array(est_xy_raw - gt_xy_raw)

    diffs_yaw = np.minimum(abs(est_yaw_raw - gt_yaw_raw),
                            abs(2. * np.pi - abs(est_yaw_raw - gt_yaw_raw))) * 180. / np.pi

    mean_square_error = np.mean(diffs_xy * diffs_xy)
    rmse_location = np.sqrt(mean_square_error)
    
    mean_square_error_yaw = np.mean(diffs_yaw * diffs_yaw)
    rmse_yaw = np.sqrt(mean_square_error_yaw)
    print('rmse_location: ', rmse_location)
    print('rmse_yaw: ', rmse_yaw)                    

if __name__ == '__main__':
    save_global_map()
    save_local_maps()
    localize()
    plot_trajectories()
    evaluation()