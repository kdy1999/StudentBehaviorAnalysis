# vim: expandtab:ts=4:sw=4

import numpy as np

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, class_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        ######该子track的序号
        self.track_id = track_id
        self.class_id = class_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        #######初始状态设定为tentative
        self.state = TrackState.Tentative
        self.features = []
        ########保存初始特征
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.increment_age()

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        ########包含每此匹配到的特征
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        ########初始化track连续匹配3此以上，则转为confirmed状态
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        #######初始化track没有连续匹配3此则删除
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        #########初始化track连续匹配3此转化为confiremed状态后，
        #########距离上此匹配间隔大于max_age则去除
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

class MyTrack(Track):
    def __init__(self,
                 attr,
                 tlwh,
                 img_id,
                 *args
                 ):
        super(MyTrack,self).__init__(*args)
        self.attr=[attr]
        self.continuity=[1]
        self.tlwh=[tlwh]
        ####符合当前的整体img_id
        self.img_ids=[img_id]
        self.start_img_id=img_id

    def update(self, kf, detection):
        super().update(kf,detection)
        ###连续性标志与属性
        self.continuity.extend([1])
        self.attr.append(detection.attr)
        self.img_ids.append(detection.img_id)
        self.tlwh.append(detection.tlwh)

    def mark_missed(self):
        super().mark_missed()
        self.continuity.append(0)
        self.attr.append([0,0,0,0,0,0,0,0,0])
        self.img_ids.append(-1)
        self.tlwh.append([0,0,0,0])


    def to_xyxy(self):
        out=np.stack(self.tlwh,axis=0)
        out[:,2]=out[:,0]+out[:,2]
        out[:, 3] = out[:, 1] + out[:, 3]
        return out




