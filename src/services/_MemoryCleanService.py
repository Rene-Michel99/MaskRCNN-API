import os
import time
import logging
import threading
from dotenv import load_dotenv


class MemoryCleanService:

    def __init__(self, images_dir: str, max_file_size: float, clean_time_window: float) -> None:
        load_dotenv()
        self.thread = None
        self.images_dir = images_dir
        self.max_file_size = max_file_size
        self.clean_time_window = clean_time_window
        self.logger = self._build_logger("./logs/memoryCleaner.log")
        self.running = False

        self.logger.info(
            f"Memory cleaner configured with images_dir: {images_dir}, max_file_size: {max_file_size}Gb and clean_time_window: {clean_time_window}s"
        )
    
    def _build_logger(self, log_dir: str):
        logger = logging.getLogger("MemoryCleaner")
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()

        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        file_handler = logging.FileHandler(
            filename=log_dir,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
    
    def start(self):
        if self.thread is None:
            self.running = True
            self.logger.info("Starting memory cleaner thread")
            self.thread = threading.Thread(target=self._clean_files)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        self.running = False
    
    def _clean_files(self):
        while self.running:
            time.sleep(self.clean_time_window)
            consumed_size = sum([
                os.path.getsize(os.path.join(self.images_dir, file_name)) / (1024 * 1024 * 1024)
                for file_name in os.listdir(self.images_dir)
            ])
            if consumed_size < self.max_file_size:
                self.logger.info(f"The memory size is {consumed_size}, no need to clean yet")
                continue

            self.logger.info("Max file size reached, starting clean")
            files_erased = 0
            locked_files = 0
            size_cleaned = 0
            for file_name in os.listdir(self.images_dir):
                if file_name.endswith(".lock") or self._file_is_locked(file_name):
                    locked_files += 1
                    continue
                
                file_path = os.path.join(self.images_dir, file_name)
                try:
                    size_cleaned += os.path.getsize(file_path) / (1024 * 1024 * 1024)
                    os.remove(file_path)
                    files_erased += 1
                except:
                    pass
            
            self.logger.info(
                "Clean ended successfully with {} files erased ({}Gb), locked files reimained {}".format(
                    files_erased, size_cleaned, locked_files
                )
            )
        
        self.logger.info("Memory cleaner thread stopped")
    
    def _file_is_locked(self, file_name: str):
        file_wt_ext = file_name.split(".")[0]
        return os.path.exists(os.path.join(self.images_dir, file_wt_ext + ".lock"))
