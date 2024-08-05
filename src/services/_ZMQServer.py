import zmq
import logging
import threading


class ZMQServer:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.running = False
        self.context = zmq.Context()
        self.router_socket = None
        self.pub_socket = None
    
    def start(self):
        th = threading.Thread(target=self._process)
        th.daemon = True
        th.start()
    
    def stop(self):
        self.running = False
        self.router_socket.close()
        self.pub_socket.close()
        self.context.term()
    
    def _process(self):
        self.running = True

        # Cria um socket ROUTER para receber mensagens dos workers
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind("tcp://localhost:5555")

        # Cria um socket PUB para enviar mensagens aos workers
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind("tcp://localhost:5556")

        self.logger.info("ZMQ Server waiting messages")
        
        while self.running:
            try:
                _, identity, msg = self.router_socket.recv_multipart()
                self.logger.info(f"Message received from {identity.decode()}: {msg.decode()}, sending for all workers")
                self.pub_socket.send_string(msg)
                self.logger.info("Message published successfully!")
            except zmq.error.ContextTerminated:
                break
            except Exception as ex:
                self.logger.exception(str(ex))