import uuid
import cv2 as cv
import numpy as np
from logging import Logger

from ..exceptions import BadRequestException
from ..models import APIConfig, ModelWrapper
from ..handlers import ImageServiceHandler


class MaskRCNNInferenceRoute:

    def __init__(self, logger: Logger, api_config: APIConfig) -> None:
        self.logger = logger
        self.api_config = api_config
        self.image_handler = ImageServiceHandler(self.api_config.images_dir)
    
    def process(self, request: dict, model: ModelWrapper) -> dict:
        self._validate_request(request)
        
        image = self.image_handler.get_image(request)
        results = model.detect([image], verbose=1)[0]

        return self._parse_detections(results, image.shape, model)
    
    def _validate_request(self, data) -> None:
        self.logger.info("Validating request")
        if 'image' not in data.keys():
            raise BadRequestException(message="no image found in request")
        if 'classes' not in data.keys():
            raise BadRequestException(message="no classes found in request")
        
        image_data = data["image"]
        if not image_data.startswith("data:image") and not image_data.startswith("http://") and \
            not image_data.startswith("https://"):
            raise BadRequestException("Image encode format not allowed, valid format are base64 (data:filename/png;base64,image_base64_data) or URL")      

    def _parse_detections(self, results, img_shape, model) -> dict:
        self.logger.info("Starting to parse results of inference")
        
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
        
        self.logger.info("Results parsed successfully, replying response")
        return output_data

    def _parse_class_ids(self, class_ids, model) -> list:
        parsed_class_ids = []
        for class_id in class_ids:
            parsed_class_ids.append(model.config.CLASS_NAMES[class_id])
        
        return parsed_class_ids

    def _parse_masks(self, masks) -> list:
        oned_masks = []
        for i in range(masks.shape[2]):
            mask = masks[:, :, i]
            contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            oned_masks.append(self._convert_mask_img_to_2d_array_contours(contours))
        
        return oned_masks
    
    def _convert_mask_img_to_2d_array_contours(self, contours) -> list:
        resized_contours = []
        for cnt in contours:
            cnt_approx = cv.approxPolyDP(cnt, self.api_config.approx_epsilon, True)
            if len(resized_contours) == 0:
                resized_contours.append(cnt_approx)
            elif len(resized_contours) == 1 and len(resized_contours[0]) < len(cnt_approx):
                resized_contours[0] = cnt_approx
        
        return resized_contours[0].reshape(-1, 2).tolist()