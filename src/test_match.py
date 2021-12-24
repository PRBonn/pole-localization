import numpy as np
import os
import matplotlib.pyplot as plt
import cluster
import scipy.special
import scipy.stats

globalmapname_gt = "globalmap_gt_cluster"
globalmapfile = os.path.join('data/pole-dataset/NCLT', globalmapname_gt + '.npz')
data = np.load(globalmapfile)

evalmapdata = np.load(os.path.join('data/pole-dataset', 'globalmap_8_0.50_0.08_108' + '.npz'))

def evaluate_matches(globalmapdata, evalmapdata):
    n = globalmapdata['poleparams'].shape[0]
    evalpolemap = evalmapdata['polemeans'][:, :2]
    n_eval = evalpolemap.shape[0]
    maxdist = 1.0
    kdtree = scipy.spatial.cKDTree(globalmapdata['poleparams'][:, :2], leafsize=10)
    dist, _ = kdtree.query(evalpolemap, k=1, distance_upper_bound=maxdist)
    n_matches = np.sum(np.isfinite(dist))

    matched_param = evalpolemap[np.isfinite(dist),:]
    TP = n_matches
    FP = n_eval - n_matches
    FN = n - n_matches
    precision = (TP+0.0)/(TP+FP)
    recall = (TP+0.0)/(TP+FN)
    F1 = 2*precision*recall/(precision+recall)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", F1)
    return matched_param

matched_param = evaluate_matches(data, evalmapdata)
