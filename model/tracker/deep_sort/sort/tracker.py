# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track,MyTrack


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for i,track in enumerate(self.tracks):
            track.predict(self.kf)



    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections, classes):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # 级联匹配
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        ####匹配track  matches [(track_idx, detection_idx)]
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        #####未匹配track
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        #####初始化track
        for detection_idx in unmatched_detections:
            ######每个tracker保存位置信息和类别信息
            self._initiate_track(detections[detection_idx], classes[detection_idx].item())
        ########删除长时间未能匹配的track
        self.tracks = [t for t in self.tracks if not t.is_deleted()]


        # Update distance metric.
        ##########confiremed track
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            ####[[feat_tk1,feat_tk1...],[feat_tk2,feat_tk2...]...]
            features += track.features
            ####[[id_tk1,id_tk1...],[id_tk2,id_tk2...]...]
            targets += [track.track_id for _ in track.features]
            track.features = []

        #########将匹配的track的特征更新为最后100个
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)


    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        ######为和只对confiremed track进行外观匹配？？？因为基于特征的匹配在刚开始的时候是不稳定的
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)


        # Associate remaining tracks together with unconfirmed tracks using IOU.
        ########将外观匹配剩下的tracker中最近匹配过目标的track与unconfiremed track结合
        #######对于初始化的tracker，在转换为confiremed之前（连续三此匹配上），只使用iou进行匹配
        ######同时对于外观未匹配上的也使用iou进行匈牙利匹配
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]

        #####confirmed_tracker中连续两次为匹配的tracker不能使用iou_tracker进行匹配，只能使用特征匹配
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]

        #########使用iou进对一阶段未匹配的track和剩余的detection匹配
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)


        ####将一阶段与二阶段匹配和未匹配的track相加
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, class_id):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, class_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1



class MyTracker(Tracker):
    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super(MyTracker,self).__init__(*args,
                                       **kwargs)

    def _initiate_track(self, detection, class_id):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(MyTrack(detection.attr,detection.tlwh,detection.img_id,
            mean, covariance, self._next_id, class_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1

    def init_tracker(self):
        del self.tracks
        self.tracks=[]
        del self.kf
        self.kf = kalman_filter.KalmanFilter()





