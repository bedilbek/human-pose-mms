import mxnet as mx
from gluoncv import model_zoo
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
from gluoncv.data.transforms.presets.yolo import transform_test


class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 0
        self.initialized = False

    def initialize(self, context):
        self.initialized = True

        # GluonCV part
        self.detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
        self.pose_net = model_zoo.get_model('alpha_pose_resnet101_v1b_coco', pretrained=True)

        # Note that we can reset the classes of the detector to only include
        # human, so that the NMS process is faster.

        self.detector.reset_class(["person"], reuse_weights=['person'])

    def preprocess(self, batch):
        img_arr = mx.img.imdecode(batch[0]['body'])
        x, img = transform_test([img_arr], short=512)

        return x, img

    def inference(self, model_input):
        x, img = model_input
        class_ids, scores, bounding_boxes = self.detector(x)
        pose_input, upscale_bbox = detector_to_alpha_pose(img, class_ids, scores, bounding_boxes)
        predicted_heatmap = self.pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

        return pred_coords, confidence

    def postprocess(self, inference_output):
        pred_coords, confidences = inference_output
        result = []
        for human_pred_coords, human_confidences in zip(pred_coords.asnumpy().tolist(), confidences.asnumpy().tolist()):
            result.append({'coords': human_pred_coords, 'confidences': human_confidences})

        return [{'estimations': result}]

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """

        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)


_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
