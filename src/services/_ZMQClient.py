import zmq
import logging
import threading

from ..handlers import BlockSystemHandler


class ZMQClient:
    def __init__(self, worker_name: str, block_system: BlockSystemHandler, logger: logging.Logger):
        self.worker_name = worker_name
        self.block_system = block_system
        self.logger = logger
        self.context = zmq.Context()
        self.running = False
        self.sub_socket = None
    
    def start_listen(self):
        th = threading.Thread(target=self._listen)
        th.daemon = True
        th.start()
    
    def stop_listen(self):
        self.running = False
        self.sub_socket.close()
        self.context.term()
    
    def send_message(self, message: str):
        # Cria um socket DEALER para enviar mensagens ao servidor
        dealer_socket = self.context.socket(zmq.DEALER)
        dealer_socket.connect("tcp://localhost:5555")

        dealer_socket.send_multipart([self.worker_name.encode(), message.encode()])
        dealer_socket.close()
    
    def _listen(self):
        self.running = True
        # Cria um socket SUB para receber mensagens do servidor
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect("tcp://localhost:5556")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        while self.running:
            try:
                msg = self.sub_socket.recv_string()

                if msg.decode() == "BLOCK_SYSTEM":
                    self.logger.info("BLOCK_SYSTEM message received, starting to block system")
                    self.block_system.block()
                    self.logger.info("Server is now blocked")
            except zmq.error.ContextTerminated:
                break
            except Exception as ex:
                self.logger.exception(str(ex))
