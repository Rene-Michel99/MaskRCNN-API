import os
import psutil


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
        
        response = {worker: "dead" for worker in self.workers.values()}
        for proc in psutil.process_iter(["pid", "name"]):
            proc_pid = str(proc.info["pid"])
            if proc_pid in self.workers:
                response[self.workers[proc_pid]] = "alive"
        
        return response
