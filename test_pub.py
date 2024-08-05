import zmq
import time

host = "localhost"
port = "5001"

# Creates a socket instance
context = zmq.Context()

dealer_socket = context.socket(zmq.DEALER)
dealer_socket.connect("tcp://localhost:5555")

time.sleep(1)

# Sends a multipart message
dealer_socket.send_multipart(["Cuca".encode(), "hello".encode()])
dealer_socket.close()