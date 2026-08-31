import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1]) + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.id_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        ret = []
        for trk in self.trackers:
            pos = trk['pos']
            trk['age'] += 1
            if trk['age'] > self.max_age:
                trk['kill'] = True
        
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, self.trackers, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]]['pos'] = dets[m[0], :4]
            self.trackers[m[1]]['age'] = 0
            self.trackers[m[1]]['hits'] += 1
            self.trackers[m[1]]['time_since_update'] = 0

        for i in unmatched_dets:
            self.id_count += 1
            self.trackers.append({'pos': dets[i, :4], 'id': self.id_count, 'age': 0, 'hits': 1, 'time_since_update': 0})

        for trk in self.trackers:
            if trk['hits'] >= self.min_hits:
                ret.append(np.concatenate((trk['pos'], [trk['id']])).reshape(1, -1))
        
        self.trackers = [t for t in self.trackers if not t.get('kill', False)]
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det, trk['pos'])
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.stack((row_ind, col_ind), axis=1)
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)