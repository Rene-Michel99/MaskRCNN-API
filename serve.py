#!/usr/bin/env python

# This file implements the scoring service shell. You don't necessarily need to modify it for various
# algorithms. It starts nginx and gunicorn with the correct configurations and then simply waits until
# gunicorn exits.
#
# The flask server is specified to be the app object in wsgi.py
#
# We set the following parameters:
#
# Parameter                Environment Variable              Default Value
# ---------                --------------------              -------------
# number of workers        MODEL_SERVER_WORKERS              the number of CPU cores
# timeout                  MODEL_SERVER_TIMEOUT              60 seconds

import os
import signal
import subprocess
import requests
import urllib.request
import logging
import shutil
import sys
import json
import socket
import random
import threading
import multiprocessing
from dotenv import load_dotenv


if os.path.exists(".env"):
    load_dotenv()

cpu_count = multiprocessing.cpu_count()

model_server_timeout = os.environ.get('SERVER_TIMEOUT', 60)
model_server_workers = int(os.environ.get('SERVER_WORKERS', cpu_count))
adjs = [
    'Saltitante', 'Cansado', 'Risonho', 'Berrante', 'Trombudo', 'Zangado', 'Pulante',
    'Fedorento', 'Choroso', 'Avexado', 'Atrevido', 'Careca', 'Calvo'
]
subts = [
    'Cavalo', 'Pinguim', 'Peixe', 'Urso', 'Lhamazinha', 'Sardinha',
    'Papagaio', 'Arara', 'Pato', 'Ursinho', 'Galinha'
]

if not os.path.exists("/app/logs"):
    os.mkdir("/app/logs")

logger = logging.getLogger("SERVE")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()

handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

file_handler = logging.FileHandler(
    filename=os.path.join("/app/logs/serve.log"),
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def sigterm_handler(nginx_pid, gunicorn_pid):
    try:
        os.kill(nginx_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(gunicorn_pid, signal.SIGTERM)
    except OSError:
        pass

    sys.exit(0)

def start_server():
    logger.info('Starting the inference server with {} workers.'.format(model_server_workers))

    # link the log streams to stdout/err so they will be logged to the container logs
    subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
    subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])

    nginx = subprocess.Popen(['nginx', '-c', '/app/nginx.conf'])
    gunicorn = subprocess.Popen(['gunicorn',
                                 '--timeout', str(model_server_timeout),
                                 '-k', 'sync',
                                 '-b', 'unix:/tmp/gunicorn.sock',
                                 '-w', str(model_server_workers),
                                 'wsgi:app'])

    signal.signal(signal.SIGTERM, lambda a, b: sigterm_handler(nginx.pid, gunicorn.pid))

    # Exit the inference server upon exit of either subprocess
    pids = set([nginx.pid, gunicorn.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break

    sigterm_handler(nginx.pid, gunicorn.pid)
    logger.info('Inference server exiting')


def download_dependencies():
    logger.info("Downloading weights for models...")
    config = {}
    with open("./config.json", "r") as f:
        config = json.loads(f.read())

    if not os.path.exists("logs/weights"):
        os.system("mkdir logs/weights")
    if not os.path.exists("./images"):
        os.system("mkdir images")

    index = 0
    total = len(config["weights"])
    for weight in config["weights"]:
        logger.info("Starting to download {} [{}/{}]".format(weight["name"], index + 1, total))
        file_path = os.path.join("logs", "weights", "{}.h5".format(weight["name"]))
        
        if os.path.exists(file_path):
            continue
        
        if weight.get("requestType", "fileTransfer") == "fileTransfer":
            download_file(weight["url"], file_path)
        else:
            logger.info("Url pre assigned found, sending request to get file url")
            url = request_file_url(weight["url"])
            download_file(url, file_path)
        
        logger.info("{} weight downloaded! [{}/{}]".format(weight["name"], index + 1, total))
        index += 1
    
    logger.info("All weights downloaded!")


def download_file(url, file_path):
    with urllib.request.urlopen(url) as resp, open(file_path, 'wb') as out:
        shutil.copyfileobj(resp, out)


def request_file_url(signed_url):
    response = requests.get(signed_url)
    if response.status_code == 200:
        logger.info("Weights file url got successfully, starting download")
        data = json.loads(response.content.decode())
        
        return data["body"]
    
    raise Exception(f"Invalid url {signed_url}")


def wait_to_send_worker_names():
    random.seed(42)
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(('localhost', 3000))
    server_sock.listen(model_server_workers)
    logger.info("Waiting connection from workers...")

    for _ in range(model_server_workers):
        conn, _ = server_sock.accept()
        worker_name = random.choice(subts) + random.choice(adjs)
        logger.info("Connection received! Replying with worker id {}".format(worker_name))
        conn.sendall(worker_name.encode())
        conn.close()
    server_sock.close()


def start_worker_namer():
    th = threading.Thread(target=wait_to_send_worker_names)
    th.daemon = True
    th.start()


# The main routine to invoke the start function.

if __name__ == '__main__':
    start_worker_namer()
    download_dependencies()
    start_server()