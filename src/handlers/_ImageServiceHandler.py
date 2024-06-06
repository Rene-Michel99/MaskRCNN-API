import os
import uuid
import base64
import shutil
import cv2 as cv
import urllib.request

from ..exceptions import UnprocessableRequest, BadRequestException


class ImageServiceHandler:

    def __init__(self, image_dir: str):
        self.image_dir = image_dir
    
    def get_image(self, data: dict):
        image_data = data["image"]
        if image_data.startswith("data:image"):
            return self._parse_base64_image(image_data)
        elif image_data.startswith("http://") or image_data.startswith("https://"):
            return self._download_image(image_data)
    
    def _download_image(self, image_url: str):
        response = urllib.request.urlopen(image_url)
        if response.code != 200:
            raise UnprocessableRequest(f"URL replied with status code {response.code}")
        
        file_extension = response.headers.get_content_type()
        if not file_extension.startswith("image"):
            raise BadRequestException(f"URL {image_url} replied with a non image file type {file_extension}")

        file_name = str(uuid.uuid4())
        file_extension = file_extension.replace("image/", "")
        image_path = os.path.join(self.image_dir, f"{file_name}.{file_extension}")
        self._lock_file(file_name)
        try:
            with open(image_path, 'wb') as out:
                shutil.copyfileobj(response, out)
        except:
            self._unlock_file(file_name)
            if os.path.exists(image_path):
                os.remove(image_path)
            
            raise UnprocessableRequest(f"Can't download image from url {image_url}")
        
        image = cv.imread(image_path)
        self._unlock_file(file_name)
        return image
    
    def _lock_file(self, file_name: str):
        with open(os.path.join(self.image_dir, file_name + ".lock"), "w") as f:
            f.write("lock")
    
    def _unlock_file(self, file_name: str):
        os.remove(os.path.join(self.image_dir, file_name + ".lock"))
    
    def _parse_base64_image(self, image_data: str):
        image_path = ""
        try:
            file_ext, encoded_image = image_data.split(',')

            file_ext = file_ext.replace('data:image', '')
            file_ext = file_ext.replace('/', '.')
            file_ext = file_ext.replace(';base64', '')
            file_name = str(uuid.uuid4()) + file_ext
            
            image_path = os.path.join(self.image_dir, file_name)
            with open(image_path, 'wb') as f:
                f.write(base64.b64decode(encoded_image))
            
            return cv.imread(image_path)
        except ValueError:
            raise BadRequestException(
                 "The image data must be encoded in base64 with pattern data:filename/png;base64,image_base64_data"
            )
        except Exception:
            if os.path.exists(image_path):
                os.remove(image_path)
            
            raise UnprocessableRequest("image can't be used, maybe is corrupted")
    