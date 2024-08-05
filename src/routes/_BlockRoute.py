from logging import Logger

from ..services import ZMQClient
from ..handlers import BlockSystemHandler


class BlockRoute:
    def __init__(self, zmq_client: ZMQClient, logger: Logger) -> None:
        self.zmq_client = zmq_client
        self.logger = logger
    
    def process(self):
        self.logger.info("Block request received, blocking system...")
        self.zmq_client.block_system.block()
        self.zmq_client.send_message("BLOCK_SYSTEM")
        self.zmq_client.stop_listen()
        
        self.logger.info("System is blocked!")
        return {}

