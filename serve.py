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

import multiprocessing
import os
import signal
import subprocess
import urllib.request
import shutil
import sys
import json
from dotenv import load_dotenv


if os.path.exists(".env"):
    load_dotenv()

cpu_count = multiprocessing.cpu_count()

model_server_timeout = os.environ.get('SERVER_TIMEOUT', 60)
model_server_workers = int(os.environ.get('SERVER_WORKERS', cpu_count))

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
    print('Starting the inference server with {} workers.'.format(model_server_workers))

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
    print('Inference server exiting')


def download_dependencies():
    print("Downloading weights for models...")
    config = {}
    with open("./config.json", "r") as f:
        config = json.loads(f.read())

    if not os.path.exists("logs"):
        os.system("mkdir logs")
        os.system("mkdir logs/weights")
    if not os.path.exists("./images"):
        os.system("mkdir images")

    for file_name, url in config["weights"].items():
        file_path = os.path.join("logs", "weights", file_name)
        with urllib.request.urlopen(url) as resp, open(file_path, 'wb') as out:
            shutil.copyfileobj(resp, out)
    
    print("All weights downloaded!")


# The main routine to invoke the start function.

if __name__ == '__main__':
    download_dependencies()
    start_server()