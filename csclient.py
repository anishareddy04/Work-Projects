import socket
import threading
import ssl

class LeastResponseTime:
    def __init__(self, servers):
        self.servers = servers
        self.response_times = {server: float('inf') for server in servers}
        self.lock = threading.Lock()

    def get_server(self):
        with self.lock:
            server = min(self.response_times, key=self.response_times.get)
            return server

    def update_response_time(self, server, response_time):
        with self.lock:
            self.response_times[server] = response_time

def client(id, least_response_time):
    try:
        server = least_response_time.get_server()
        host, port = server
        
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            with context.wrap_socket(s, server_hostname=host) as client_socket:
                client_socket.connect((host, port))
                print(f"Client {id} connected to server {host}:{port}. Type 'quit' to exit.")

                while True:
                    message = input(f"Client {id} You: ")
                    if message.lower() == 'quit':
                        break
                    client_socket.sendall(message.encode())
                    response = client_socket.recv(1024).decode()

                    # Extract response time from the server response
                    try:
                        response_message, timing_info = response.rsplit(" (Running Time: ", 1)
                        response_time_str, _ = timing_info.split(", Load Balancing Time: ")
                        response_time = float(response_time_str.replace("s", "").strip())
                    except ValueError as e:
                        print(f"Client {id} encountered an error parsing the response time: {e}")
                        response_message = response
                        response_time = float('inf')

                    print(f"Client {id} ChatBot: {response_message}")
                    print(f"Client {id} Response Time: {response_time:.4f} seconds")

                    # Update the response time for the server
                    least_response_time.update_response_time(server, response_time)

                print(f"Client {id} exiting.")
    except Exception as e:
        print(f"Client {id} encountered an error: {e}")

def start_multiple_clients(num_clients, servers):
    least_response_time = LeastResponseTime(servers)
    threads = []
    for i in range(num_clients):
        thread = threading.Thread(target=client, args=(i+1, least_response_time))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    servers = [
        ('127.0.0.1', 8443),
        ('127.0.0.1', 8444),
        ('127.0.0.1', 8445)
        # Add more servers as needed
    ]
    start_multiple_clients(100, servers)  # Adjust the number of clients as needed
