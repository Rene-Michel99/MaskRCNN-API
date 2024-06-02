import os
import uuid
import base64
import cv2 as cv
import numpy as np
from logging import Logger

from ..exceptions import UnprocessableRequest, BadRequestException
from ..models import APIConfig, ModelWrapper


class MaskRCNNInferenceRoute:

    def __init__(self, logger: Logger, api_config: APIConfig) -> None:
        self.logger = logger
        self.api_config = api_config
    
    def process(self, request: dict, model: ModelWrapper):
        self._validate_request(request)
        
        image_path = self._parse_request(request)
        image = cv.imread(image_path)
        results = model.detect([image], verbose=1)[0]

        return self._parse_detections(results, image.shape, model)
    
    def _validate_request(self, data):
        if 'image' not in data.keys():
            raise BadRequestException(message="no image found in request")
        if 'classes' not in data.keys():
            raise BadRequestException(message="no classes found in request")
    
    def _parse_request(self, data: dict):
        image_dir = self.api_config.images_dir
        image_path = None
        try:
            file_ext, encoded_image = data['image'].split(',')

            file_ext = file_ext.replace('data:image', '')
            file_ext = file_ext.replace('/', '.')
            file_ext = file_ext.replace(';base64', '')
            file_name = str(uuid.uuid4()) + file_ext
            
            image_path = os.path.join(image_dir, file_name)
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(encoded_image))
            
            test_img = cv.imread(image_path)
        except ValueError as ex:
            self.logger.exception(ex)
            raise BadRequestException(
                 "The image data must be encoded in base64 with pattern data:filename/png;base64,image_base64_data"
            )
        except Exception as ex:
             self.logger.exception(ex)
             raise UnprocessableRequest("image can't be used, maybe is corrupted")
            
        return image_path

    def _parse_detections(self, results, img_shape, model):
        self.logger.info("starting to parse results of inference")
        
        rois = results["rois"].tolist() if type(results["rois"]) != list else results["rois"]
        class_ids = self._parse_class_ids(results["class_ids"], model)
        scores = results["scores"].tolist() if type(results["scores"]) != list else results["scores"]
        masks = self._parse_masks(results["masks"])

        output_data = {
            'inferences': [],
            'imgSize': img_shape
        }
        for roi, class_id, score, mask in zip(rois, class_ids, scores, masks):
            output_data['inferences'].append({
                'id': str(uuid.uuid4()),
                'bbox': roi,
                'className': class_id,
                'score': score,
                'points': mask
            })
        
        return output_data

    def _parse_class_ids(self, class_ids, model):
        parsed_class_ids = []
        for class_id in class_ids:
            parsed_class_ids.append(model.config.CLASS_NAMES[class_id])
        
        return parsed_class_ids

    def _parse_masks(self, masks):
        oned_masks = []
        for i in range(masks.shape[2]):
            mask = masks[:, :, i]
            if np.sum(mask) > 0:
                contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                oned_masks.append(self._convert_mask_img_to_2d_array_contours(contours))
        
        return oned_masks
    
    def _convert_mask_img_to_2d_array_contours(self, contours):
        oned_contours = []
        for cnt in contours:
            cnt_approx = cv.approxPolyDP(cnt, self.api_config.approx_epsilon, True)
            for point in cnt_approx:
                oned_contours.append(point[0].tolist())
        
        return oned_contours
