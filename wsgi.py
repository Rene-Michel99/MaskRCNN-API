import os
import time
import socket
import logging

import src.app as myapp

# This is just a simple wrapper for gunicorn to find your app.

class WorkerBuilder:

    def __init__(self) -> None:
        self.logger = logging.getLogger("WorkerBuilder")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def build(self):
        self.logger.info("Starting creation of APIServer worker")
        
        worker_name = self._get_worker_name()
        log_dir = os.path.join(
            "/app/logs",
            worker_name,
        )
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        return myapp.APIServer(worker_name=worker_name, log_dir=log_dir)
    
    def _get_worker_name(self, timeout=5):
        self.logger.info("Connecting to main server to get id...")

        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect(('localhost', 3000))
        worker_name = None
        retries = 0
        while True:    
            try:
                msg = client_sock.recv(1024)
                worker_name = msg.decode("utf-8")
                break
            except Exception as ex:
                self.logger.exception(ex)
                retries += 1
                time.sleep(1)
            
            if retries >= timeout:
                break
        
        client_sock.close()
        self.logger.info(f"Worker name received! Creating APIServer worker {worker_name}")
        return worker_name


builder = WorkerBuilder()
api_server = builder.build()
app = api_server.app
