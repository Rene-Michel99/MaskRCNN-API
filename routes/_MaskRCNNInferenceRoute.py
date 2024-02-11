import os
import math
import base64
import cv2 as cv
import numpy as np
from logging import Logger
from sklearn.decomposition import PCA

from mrcnn.model import MaskRCNN
from exceptions import UnprocessableRequest


class MaskRCNNInferenceRoute(object):

    def __init__(self, configs, weights, logger: Logger) -> None:
        self.configs = configs
        self.weights = weights
        self.logger = logger
    
    def process(self, request: dict, model: MaskRCNN):
        self._validate_request(request)

        classes = request['classes']
        #model = self._load_proper_weights_to_model(classes)
        
        image_path = self._parse_request(request)
        image = cv.imread(image_path)
        results = model.detect([image], verbose=1)[0]

        return self._parse_detections(results, model)
    
    def _validate_request(self, data):
        if 'image' not in data.keys():
            raise UnprocessableRequest(message="no image found in request", error_code=422)
        if 'classes' not in data.keys():
            raise UnprocessableRequest(message="no classes found in request", error_code=422)
    
    def _parse_request(self, data: dict):
        image_dir = "./tests/images"
        image_path = None
        try:
            file_name, encoded_image = data['image'].split(',')

            file_name = file_name.replace('data:', '')
            file_name = file_name.replace('/', '.')
            file_name = file_name.replace(';base64', '')
            
            image_path = os.path.join(image_dir, file_name)
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(encoded_image))
        except Exception as ex:
             self.logger.exception(ex)
             raise UnprocessableRequest(message="image can't be used, maybe is corrupted", error_code=422)
            
        return image_path

    def _parse_detections(self, results, model):
            self.logger.info("starting to parse results of inference")
            rois = results["rois"].tolist() if type(results["rois"]) != list else results["rois"]
            class_ids = self._parse_class_ids(results["class_ids"], model)
            scores = results["scores"].tolist() if type(results["scores"]) != list else results["scores"]
            masks, measurements = self._parse_masks(results["masks"])

            output_data = {'inferences': []}
            for roi, class_id, score, mask, measurement in zip(rois, class_ids, scores, masks, measurements):
                output_data['inferences'].append({
                    'bbox': roi,
                    'className': class_id,
                    'score': score,
                    'points': mask,
                    'measurement': {
                        'line_points': [measurement[0], measurement[1]],
                        'size': measurement[2]
                    }
                })
            
            return output_data

    def _parse_class_ids(self, class_ids, model):
            parsed_class_ids = []
            for class_id in class_ids:
                parsed_class_ids.append(model.config.CLASS_NAMES[class_id])
            
            return parsed_class_ids

    def _parse_masks(self, masks):
            oned_masks = []
            measurements = []
            for i in range(masks.shape[2]):
                mask = masks[:, :, i]
                if np.sum(mask) > 0:
                    contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    oned_masks.append(self._convert_mask_img_to_2d_array_contours(contours))
                    measurements.append(self._calc_longest_diagonal_pca(contours))
            return oned_masks, measurements
    
    def _convert_mask_img_to_2d_array_contours(self, contours):
        oned_contours = []
        for cnt in contours:
            for point in cnt:
                oned_contours.append(point[0].tolist())
        
        return oned_contours

    def _calc_longest_diagonal_pca(self, contours):
        contour = sorted(contours, key=lambda cnt: len(cnt), reverse=True)
        contour = np.squeeze(contour[0])

        if (len(contour.shape) == 1):
            return tuple(contour), tuple(contour), 0
        
        pca = PCA(n_components=1)
        pca.fit(contour)

        principal_component = pca.components_[0]
        contour_pca = np.dot(contour, principal_component)

        start_index = np.argmin(contour_pca)
        end_index = np.argmax(contour_pca)

        start, end = contour[start_index].tolist(), contour[end_index].tolist()
        start, end = tuple(start), tuple(end)
        length = math.dist(start, end)

        return [start, end, length]
