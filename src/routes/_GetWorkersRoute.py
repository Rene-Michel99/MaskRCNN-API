import os
import time
import psutil
import datetime


class GetWorkersRoute:

    def __init__(self):
        self.workers = {}
    
    def process(self):
        if not self.workers:
            workers = os.listdir("/app/logs")
            workers.remove("weights")
            workers.remove("serve.log")
            for worker in workers:
                with open(f"/app/logs/{worker}/pid", "r") as f:
                    pid = f.read().replace("\n", "").strip()
                    self.workers[pid] = worker
        
        response = {worker: {"status": "dead"} for worker in self.workers.values()}
        for proc in psutil.process_iter(["pid", "status"]):
            proc_pid = str(proc.pid)
            if proc_pid in self.workers:
                creation_time = datetime.datetime.strptime(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(proc.create_time())),
                    '%Y-%m-%d %H:%M:%S',
                )
                time_running = datetime.datetime.now() - creation_time
                mem_usage = proc.memory_info().rss / (1024 * 1024)
                response[self.workers[proc_pid]] = {
                    "status": proc.info["status"],
                    "createdAt": creation_time.isoformat(),
                    "timeRunning": str(time_running),
                    "memUsage": mem_usage
                }
        
        return response
