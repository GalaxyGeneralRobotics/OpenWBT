import socket
import json
import numpy as np
import time
import threading
class IKClient:
    def __init__(self, host='localhost', port=12345, timeout=1.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.connected = False
        assert self.connect()
        self.latest_ik_result = None
        
    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to IK server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to IK server: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        self.connected = False
        print("Disconnected from IK server")
    
    def solve_ik_remote(self, left_wrist, right_wrist, current_lr_arm_q, current_lr_arm_dq):
        if not self.connected:
            return current_lr_arm_q
        
        try:
            request_data = {
                'left_wrist': left_wrist.tolist(),
                'right_wrist': right_wrist.tolist(),
                'current_lr_arm_q': current_lr_arm_q.tolist(),
                'current_lr_arm_dq': current_lr_arm_dq.tolist()
            }
            
            self.send_data(request_data)
            
            self.latest_ik_result = self.receive_data()
            
            if self.latest_ik_result['success']:
                return self.latest_ik_result['sol_q']
            else:
                return current_lr_arm_q
                
        except Exception as e:
            print(f"Error in remote IK call: {e}")
            self.connected = False
            return current_lr_arm_q
    
    def send_data(self, data):
        if not self.socket:
            raise ConnectionError("Not connected to server")
            
        json_str = json.dumps(data)
        data_bytes = json_str.encode('utf-8')
        
        self.socket.send(len(data_bytes).to_bytes(4, byteorder='big'))
        self.socket.send(data_bytes)
    
    def receive_data(self):
        if not self.socket:
            raise ConnectionError("Not connected to server")
            
        try:
            length_data = self.socket.recv(4)
            if not length_data:
                return None
                
            data_length = int.from_bytes(length_data, byteorder='big')
            
            data = b''
            while len(data) < data_length:
                chunk = self.socket.recv(data_length - len(data))
                if not chunk:
                    return None
                data += chunk
            
            return json.loads(data.decode('utf-8'))
            
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None
