import datetime
import os
import arrow
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import scipy.interpolate
import cluster
import kittiwrapper
import particlefilter
import util
import poles_extractor

np.random.seed(0)
dataset = kittiwrapper.kittiwrapper('kitti')

mapextent = np.array([30.0, 30.0, 3.0])
mapsize = np.full(3, 0.1)
mapshape = np.array(mapextent / mapsize, dtype=np.int)
mapinterval = 1.5
mapdistance = 1.5

T_mc_cam0 = np.identity(4)
T_mc_cam0[:3, :3] \
    = [[0.0,  0.0, 1.0], [-1.0,  0.0, 0.0], [0.0, -1.0, 0.0]]
T_cam0_mc = util.invert_ht(T_mc_cam0)
T_m_mc = np.identity(4)
T_m_mc[:3, 3] = np.hstack([0.5 * mapextent[:2], 2.0])
T_mc_m = util.invert_ht(T_m_mc)
T_cam0_m = T_cam0_mc.dot(T_mc_m)

globalmapfile = 'globalmap_3.npz'
localmapfile = 'localmaps_3.npz'
locfileprefix = 'localization'
evalfile = 'evaluation.npz'


def get_map_indices(sequence):
    distance = np.hstack([0.0, np.cumsum(np.linalg.norm(
        np.diff(sequence.poses[:, :3, 3], axis=0), axis=1))])
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
    return istart, imid, iend


def save_local_maps(seq):
    print(seq)
    sequence = dataset.sequence(seq)
    seqdir = os.path.join('kitti', '{:03d}'.format(seq))
    util.makedirs(seqdir)
    istart, imid, iend = get_map_indices(sequence)
    maps = []
    with progressbar.ProgressBar(max_value=len(iend)) as bar:
        for i in range(len(iend)):
            T_m_w = T_m_mc.dot(T_mc_cam0).dot(
                util.invert_ht(sequence.poses[imid[i]]))
            T_w_velo = np.matmul(
                sequence.poses[imid[i]], sequence.calib.T_cam0_velo)
            T_m_velo = np.matmul(T_m_w, T_w_velo)

            iscan = imid[i]
            velo = sequence.get_velo(iscan)
            poleparams = poles_extractor.detect_poles(velo, dis_thr = 0.2, fov_up=3.0, fov_down=-25, proj_H = 64, proj_W = 500, lowest=-1.3)
            localpoleparam_xy = poleparams[:, :2]
            localpoleparam_xy = localpoleparam_xy.T
            localpoleparam_xy = np.vstack([localpoleparam_xy, np.zeros_like(localpoleparam_xy[0]), np.ones_like(localpoleparam_xy[0])]) #4*n
            localpoleparam_xy = np.matmul(T_m_velo, localpoleparam_xy)
            poleparams[:, :2] = localpoleparam_xy[:2,:].T

            map = {'poleparams': poleparams, 
                'istart': istart[i], 'imid': imid[i], 'iend': iend[i]}
            maps.append(map)
            bar.update(i)
    np.savez(os.path.join(seqdir, localmapfile), maps=maps)


def save_global_map(seq):
    sequence = dataset.sequence(seq)
    seqdir = os.path.join('kitti', '{:03d}'.format(seq))
    util.makedirs(seqdir)
    istart, imid, iend = get_map_indices(sequence)
    poleparams = np.empty([0, 3])
    with np.load(os.path.join(seqdir, localmapfile), allow_pickle=True) as data:
        for i, map in enumerate(data['maps']):
            T_w_m = sequence.poses[map['imid']].dot(T_cam0_mc).dot(T_mc_m)
            localpoleparams = map['poleparams']

            localpoleparam_xy = localpoleparams[:, :2]
            localpoleparam_xy = localpoleparam_xy.T
            localpoleparam_xy = np.vstack([localpoleparam_xy, np.zeros_like(localpoleparam_xy[0]), np.ones_like(localpoleparam_xy[0])]) #4*n
            localpoleparam_xy = np.matmul(T_w_m, localpoleparam_xy)
            localpoleparams[:, :2] = localpoleparam_xy[:2,:].T
            poleparams = np.vstack([poleparams, localpoleparams])
    
    xy = poleparams[:, :2]
    a = poleparams[:, [2]]
    boxes = np.hstack([xy - a, xy + a])
    clustermeans = np.zeros([0, 3])
    clustercovs = np.zeros([0, 3, 3])
    for ci in cluster.cluster_boxes(boxes):
        ci = list(ci)
        if len(ci) < 1:
            continue
        clustermeans = np.vstack([clustermeans, np.average(
            poleparams[ci, :], axis=0)])
    np.savez(os.path.join(seqdir, globalmapfile), polemeans=clustermeans,
        polecovs=clustercovs)


def view_global_map(seq):
    seqdir = '{:03d}'.format(seq)
    with np.load(os.path.join('kitti', seqdir, globalmapfile), allow_pickle=True) as data:
        xy = data['polemeans'][:, :2]
        plt.scatter(xy[:, 0], xy[:, 1], s=5, c='b', marker='s')
        plt.show()


def localize(seq, visualize=False):
    print(seq)
    sequence = dataset.sequence(seq)
    seqdir = os.path.join('kitti', '{:03d}'.format(seq))
    mapdata = np.load(os.path.join(seqdir, globalmapfile), allow_pickle=True)
    polemap = mapdata['polemeans']
    locdata = np.load(os.path.join(seqdir, localmapfile), allow_pickle=True)['maps']
    T_velo_cam0 = util.invert_ht(sequence.calib.T_cam0_velo)
    T_w_velo_gt = np.matmul(sequence.poses, sequence.calib.T_cam0_velo)
    i = 0
    polecov = 1.0
    filter = particlefilter.particlefilter(
        2000, T_w_velo_gt[i], 3.0, np.radians(5.0), polemap, polecov)
    filter.minneff = 0.5
    filter.estimatetype = 'best'

    if visualize:
        plt.ion()
        figure = plt.figure()
        nplots = 2
        mapaxes = figure.add_subplot(nplots, 1, 1)
        mapaxes.set_aspect('equal')
        mapaxes.scatter(polemap[:, 0], polemap[:, 1], s=5, c='b', marker='s')
        mapaxes.plot(T_w_velo_gt[:, 0, 3], T_w_velo_gt[:, 1, 3], 'g')
        particles = mapaxes.scatter([], [], s=1, c='r')
        arrow = mapaxes.arrow(0.0, 0.0, 3.0, 0.0, length_includes_head=True, 
            head_width=2.1, head_length=3.0, color='k')
        arrowdata = np.hstack(
            [arrow.get_xy(), np.zeros([8, 1]), np.ones([8, 1])]).T
        locpoles = mapaxes.scatter([], [], s=30, c='k', marker='x')
        viewoffset = 25.0

        weightaxes = figure.add_subplot(nplots, 1, 2)
        gridsize = 50
        offset = 10.0
        visfilter = particlefilter.particlefilter(gridsize**2, 
            np.identity(4), 0.0, 0.0, polemap, polecov)
        gridcoord = np.linspace(-offset, offset, gridsize)
        x, y = np.meshgrid(gridcoord, gridcoord)
        dxy = np.hstack([x.reshape([-1, 1]), y.reshape([-1, 1])])
        weightimage = weightaxes.matshow(np.zeros([gridsize, gridsize]), 
            extent=(-offset, offset, -offset, offset))

    imap = 0
    while locdata[imap]['imid'] < i:
        imap += 1
    T_w_velo_est = np.full(T_w_velo_gt.shape, np.nan)
    T_w_velo_est[i] = filter.estimate_pose()
    i += 1
    with progressbar.ProgressBar(max_value=T_w_velo_est.shape[0] - i) as bar:
        while i < T_w_velo_est.shape[0]:
            relodo = util.ht2xyp(
                util.invert_ht(T_w_velo_gt[i-1]).dot(T_w_velo_gt[i]))
            relodocov = np.diag((0.02 * relodo)**2)
            relodo = np.random.multivariate_normal(relodo, relodocov)
            filter.update_motion(relodo, relodocov)
            T_w_velo_est[i] = filter.estimate_pose()

            if imap < locdata.size and i >= locdata[imap]['iend']:
                T_w_cam0_mid = sequence.poses[locdata[imap]['imid']]
                T_w_cam0_now = sequence.poses[i]
                T_cam0_now_cam0_mid \
                    = util.invert_ht(T_w_cam0_now).dot(T_w_cam0_mid)

                poleparams = locdata[imap]['poleparams']
                npoles = poleparams.shape[0]
                h = np.diff(poleparams[:, 2:4], axis=1)
                polepos_m_mid = np.hstack([poleparams[:, :2], 
                    np.zeros([npoles, 1]), np.ones([npoles, 1])]).T
                polepos_velo_now = T_velo_cam0.dot(T_cam0_now_cam0_mid).dot(
                    T_cam0_m).dot(polepos_m_mid)
                poleparams = polepos_velo_now[:2].T
                filter.update_measurement(poleparams[:, :2])
                T_w_velo_est[i] = filter.estimate_pose()
                if visualize:
                    polepos_w = T_w_velo_est[i].dot(polepos_velo_now)
                    locpoles.set_offsets(polepos_w[:2].T)

                    particleposes = np.tile(T_w_velo_gt[i], [gridsize**2, 1, 1])
                    particleposes[:, :2, 3] += dxy
                    visfilter.particles = particleposes
                    visfilter.weights[:] = 1.0 / visfilter.count
                    visfilter.update_measurement(poleparams[:, :2], resample=False)
                    weightimage.set_array(np.flipud(
                        visfilter.weights.reshape([gridsize, gridsize])))
                    weightimage.autoscale()
                imap += 1

            if visualize:
                particles.set_offsets(filter.particles[:, :2, 3])
                arrow.set_xy(T_w_velo_est[i].dot(arrowdata)[:2].T)
                x, y = T_w_velo_est[i, :2, 3]
                mapaxes.set_xlim(left=x - viewoffset, right=x + viewoffset)
                mapaxes.set_ylim(bottom=y - viewoffset, top=y + viewoffset)
                figure.canvas.draw_idle()
                figure.canvas.flush_events()
            bar.update(i)
            i += 1
    filename = os.path.join(seqdir, locfileprefix \
        + datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.npz'))
    np.savez(filename, T_w_velo_est=T_w_velo_est)


def evaluate(seq):
    sequence = dataset.sequence(seq)
    T_w_velo_gt = np.matmul(sequence.poses, sequence.calib.T_cam0_velo)
    T_w_velo_gt = np.array([util.project_xy(ht) for ht in T_w_velo_gt])
    seqdir = os.path.join('kitti', '{:03d}'.format(seq))
    mapdata = np.load(os.path.join(seqdir, globalmapfile), allow_pickle=True)
    polemap = mapdata['polemeans']
    plt.scatter(polemap[:, 0], polemap[:, 1], s=1, c='b')
    plt.plot(T_w_velo_gt[:, 0, 3], T_w_velo_gt[:, 1, 3], color=(0.5, 0.5, 0.5))
    cumdist = np.hstack([0.0, np.cumsum(np.linalg.norm(np.diff(
        T_w_velo_gt[:, :2, 3], axis=0), axis=1))])
    timestamps = np.array([arrow.get(timestamp).float_timestamp \
        for timestamp in sequence.timestamps])
    t_eval = scipy.interpolate.interp1d(
        cumdist, timestamps)(np.arange(0.0, cumdist[-1], 1.0))
    n = t_eval.size
    T_w_velo_gt_interp = np.empty([n, 4, 4])
    iodo = 1
    for ieval in range(n):
        while timestamps[iodo] < t_eval[ieval]:
            iodo += 1
        T_w_velo_gt_interp[ieval] = util.interpolate_ht(
            T_w_velo_gt[iodo-1:iodo+1], timestamps[iodo-1:iodo+1], 
            t_eval[ieval])
    files = [file for file in os.listdir(seqdir) \
        if os.path.basename(file).startswith(locfileprefix)]
    poserror = np.full([n, len(files)], np.nan)
    laterror = np.full([n, len(files)], np.nan)
    lonerror = np.full([n, len(files)], np.nan)
    angerror = np.full([n, len(files)], np.nan)
    T_gt_est = np.full([n, 4, 4], np.nan)
    for ifile in range(len(files)):
        T_w_velo_est = np.load(
            os.path.join(seqdir, files[ifile]), allow_pickle=True)['T_w_velo_est']
        iodo = 1
        for ieval in range(n):
            while timestamps[iodo] < t_eval[ieval]:
                iodo += 1
            T_w_velo_est_interp = util.interpolate_ht(
                T_w_velo_est[iodo-1:iodo+1], timestamps[iodo-1:iodo+1], 
                t_eval[ieval])
            T_gt_est[ieval] = util.invert_ht(T_w_velo_gt_interp[ieval]).dot(
                T_w_velo_est_interp)
        lonerror[:, ifile] = T_gt_est[:, 0, 3]
        laterror[:, ifile] = T_gt_est[:, 1, 3]
        poserror[:, ifile] = np.linalg.norm(T_gt_est[:, :2, 3], axis=1)
        angerror[:, ifile] = util.ht2xyp(T_gt_est)[:, 2]
        plt.plot(T_w_velo_est[:, 0, 3], T_w_velo_est[:, 1, 3], 'r')
    angerror = np.degrees(angerror)
    lonstd = np.std(lonerror, axis=0)
    latstd = np.std(laterror, axis=0)
    angstd = np.std(angerror, axis=0)
    angerror = np.abs(angerror)
    laterror = np.mean(np.abs(laterror), axis=0)
    lonerror = np.mean(np.abs(lonerror), axis=0)
    posrmse = np.sqrt(np.mean(poserror ** 2, axis=0))
    angrmse = np.sqrt(np.mean(angerror ** 2, axis=0))
    poserror = np.mean(poserror, axis=0)
    angerror = np.mean(angerror, axis=0)
    plt.savefig(os.path.join(seqdir, 'trajectory_est.svg'))
    np.savez(os.path.join(seqdir, evalfile), 
        poserror=poserror, angerror=angerror, posrmse=posrmse, angrmse=angrmse,
        laterror=laterror, latstd=latstd, lonerror=lonerror, lonstd=lonstd)
    print('poserror: {}\nposrmse: {}\n'
        'laterror: {}\nlatstd: {}\n'
        'lonerror: {}\nlonstd: {}\n'
        'angerror: {}\nangstd: {}\nangrmse: {}'.format(
            np.mean(poserror), np.mean(posrmse), 
            np.mean(laterror), np.mean(latstd), 
            np.mean(lonerror), np.mean(lonstd),
            np.mean(angerror), np.mean(angstd), np.mean(angrmse)))


if __name__ == '__main__':
    save_local_maps(13)
    save_global_map(13)
    localize(13, visualize=False)
    evaluate(13)