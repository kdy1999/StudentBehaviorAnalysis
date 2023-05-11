from .deep_sort.deep_sort import DeepSort
from .deep_sort.sort.detection import MyDetection
from model.tracker.deep_sort.deep.feature_extractor import Extractor
from model.tracker.deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from model.tracker.deep_sort.sort.tracker import MyTracker

class deeptracker(DeepSort):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7,
                 max_age=70, n_init=3, nn_budget=100, use_cuda=False):

        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = MyTracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
        self.img_id=0

    def update(self, bbox_xywh, confidences, classes, attr, ori_img, img_id):
        self.height, self.width = ori_img.shape[:2]
        ######提取每个目标框的特征
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [MyDetection(attr[i],img_id,bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.min_confidence]
        self.tracker.predict()
        self.tracker.update(detections, classes[confidences>self.min_confidence])


