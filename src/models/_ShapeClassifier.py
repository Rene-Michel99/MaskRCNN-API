import pickle
import skimage
import numpy as np


class ShapeClassifier:

    def __init__(self, weights_path: str, classes: list, filter_by_class_name):
        self.classes = classes
        self.filter_by_class_name = filter_by_class_name
        with open(weights_path, "rb") as f:
            self.model = pickle.load(f)
    
    def predict(self, img, class_name: str):
        if class_name != self.filter_by_class_name and self.filter_by_class_name != None:
            return None
        
        data = []
        regions = skimage.measure.regionprops(label_image=img)
        max_region = max(regions, key=lambda region: region.area)
        data.append(max_region.area)
        data.append(max_region.perimeter)
        data.append(max_region.eccentricity)
        data.append(max_region.solidity)
        data.append(max_region.extent)
        data.append(4 * np.pi * (max_region.area / (max_region.perimeter ** 2)))
        data.append(max_region.convex_area)
        data.append(max_region.equivalent_diameter)
        data.append(max_region.major_axis_length)
        data.append(max_region.minor_axis_length)

        minr, minc, maxr, maxc = max_region.bbox
        width = maxc - minc
        height = maxr - minr
        aspect_ratio = width / height if height != 0 else 0

        data.append(width)
        data.append(height)
        data.append(aspect_ratio)

        data = np.array(data).reshape(1, -1)
        class_id = self.model.predict(data)[0]
        
        return self.classes[class_id]
