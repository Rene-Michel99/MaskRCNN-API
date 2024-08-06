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
        self.image_handler = ImageServiceHandler(self.api_config.images_dir, self.logger)
    
    def process(self, request: dict, model: ModelWrapper) -> dict:
        self._validate_request(request)
        
        image = self.image_handler.get_image(request)
        images = [((0, 0), image)]
        if image.shape[0] > 1024 and image.shape[1] > 1024 and self.api_config.split_images_above_maximum:
            self.logger.info("Image received has {} which is above the maximum (1024, 1024, 3), splitting image in 4...".format(image.shape))
            height, width, _ = image.shape
            images = [
                ((0,0), image[0:height//2, 0:width//2].copy()),
                ((height//2, 0), image[height//2:height, 0:width//2].copy()),
                ((0, width//2), image[0:height//2, width//2:width].copy()),
                ((height//2, width//2), image[height//2:height, width//2:width].copy())
            ]
            self.logger.info("Image splitted successfully")

        return self._parse_detections(images, image.shape, model)
    
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

    def _parse_detections(self, images, img_shape, model) -> dict:
        self.logger.info("Starting to parse results of inference")
        
        output_data = {
            'inferences': [],
            'imgSize': img_shape
        }
        for data in images:
            adjust, image = data
            res = model.detect([image], verbose=1)[0]
            for i in range(len(res["class_ids"])):
                mask = res["masks"][:, :, i]
                bbox = res["rois"][i].tolist()
                class_name = model.config.CLASS_NAMES[res["class_ids"][i]]
                obj = {
                    'id': str(uuid.uuid4()),
                    'bbox': [bbox[0] + adjust[0], bbox[1] + adjust[1], bbox[2] + adjust[0], bbox[3] + adjust[1]],
                    'className': class_name,
                    'score': float(res["scores"][i]),
                    'points': self._parse_mask(mask, adjust),
                }
                metrics = self._get_extra_metrics(mask, model, class_name)
                obj.update(metrics)
                output_data['inferences'].append(obj)
        
        self.logger.info("Results parsed successfully, replying response")
        return output_data
    
    def _get_extra_metrics(self, mask, model: ModelWrapper, class_name: str):
        return model.get_extra_metrics(mask.astype(np.uint8) * 255, class_name)

    def _parse_mask(self, mask, adjust) -> list:
        contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        return self._convert_mask_img_to_2d_array_contours(contours, adjust)
    
    def _convert_mask_img_to_2d_array_contours(self, contours, adjust) -> list:
        resized_contours = []
        for cnt in contours:
            cnt_approx = cv.approxPolyDP(cnt, self.api_config.approx_epsilon, True)
            if len(resized_contours) == 0:
                resized_contours.append(cnt_approx)
            elif len(resized_contours) == 1 and len(resized_contours[0]) < len(cnt_approx):
                resized_contours[0] = cnt_approx
        
        resized_contours = resized_contours[0].reshape(-1, 2).tolist()
        if adjust[0] == 0 and adjust[1] == 0:
            return resized_contours
        
        return [[point[0] + adjust[1], point[1] + adjust[0]] for point in resized_contours]
