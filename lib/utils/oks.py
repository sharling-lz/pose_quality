from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def computeOKS(dts, gts, area, bbox):
    ious = np.zeros((len(dts), 1),dtype=np.float32)
    num_points = np.zeros((len(dts), 1))
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
    vars = (sigmas * 2)**2
    k = len(sigmas)
     # compute oks between each detection and ground truth object
    for j in range(gts.shape[0]):
        g = gts[j,:,:]
        d = dts[j,:,:]
        # create bounds for ignore regions(double the gt bbox)
        xg = g[:,0]; yg = g[:,1]; vg = g[:,2]
        k1 = np.count_nonzero(vg > 0)
        bb = bbox[j,:]
        a = area[j,:]
        x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
        y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
        
        xd = d[:,0]; yd = d[:,1]
        if k1>0:
            # measure the per-keypoint distance if keypoints visible
            dx = xd - xg
            dy = yd - yg
        else:
            # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
            z = np.zeros((k))
            dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
            dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)
        e = (dx**2 + dy**2) / vars / (a+np.spacing(1)) / 2
        if k1 > 0:
            e=e[vg > 0]
        ious[j] = np.sum(np.exp(-e)) / e.shape[0]
        num_points[j] = k1
    return ious