import zmq

host = "localhost"
port = "5001"

# Creates a socket instance
context = zmq.Context()
router_socket = context.socket(zmq.ROUTER)
router_socket.bind("tcp://localhost:5555")

# Cria um socket PUB para enviar mensagens aos workers
pub_socket = context.socket(zmq.PUB)
pub_socket.bind("tcp://localhost:5556")
# Receives a multipart message
_, x, y = router_socket.recv_multipart() 
print(x.decode(), y.decode())