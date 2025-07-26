import socket
import json
import numpy as np
import sys
import os

from deploy.teleop.robot_control.robot_arm_ik import G1_29_ArmIK

class IKServer:
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.arm_ik = None
        
        print("Initializing G1_29_ArmIK...")
        self.arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)
        print("IK solver initialized successfully!")
        self.result = {
                'sol_q': None,
                'success': False
            }
        
    def start_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            print(f"IK Server started on {self.host}:{self.port}")
            print("Waiting for IsaacSim client connection...")
            
            while True:
                try:
                    conn, addr = self.socket.accept()
                    print(f"Connected by {addr}")
                    
                    while True:
                        data = self.receive_data(conn)
                        if data is None:
                            break
                            
                        self.send_data(conn, self.result)

                        self.result = self.process_ik_request(data)
                        
                except ConnectionResetError:
                    print("Client disconnected")
                    continue
                except Exception as e:
                    print(f"Error handling client: {e}")
                    continue
                finally:
                    if 'conn' in locals():
                        conn.close()
                        
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            if self.socket:
                self.socket.close()
    
    def receive_data(self, conn):
        try:
            length_data = conn.recv(4)
            if not length_data:
                return None
                
            data_length = int.from_bytes(length_data, byteorder='big')
            data = b''
            while len(data) < data_length:
                chunk = conn.recv(data_length - len(data))
                if not chunk:
                    return None
                data += chunk
            
            json_data = json.loads(data.decode('utf-8'))
            
            request_data = {
                'left_wrist': np.array(json_data['left_wrist']),
                'right_wrist': np.array(json_data['right_wrist']),
                'current_lr_arm_q': np.array(json_data['current_lr_arm_q']),
                'current_lr_arm_dq': np.array(json_data['current_lr_arm_dq'])
            }
            
            return request_data
            
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None
    
    def send_data(self, conn, data):
        json_data = {
            'sol_q': data['sol_q'].tolist() if data['sol_q'] is not None else None,
            'success': data['success'],
            'error_msg': data.get('error_msg', '')
        }
        
        json_str = json.dumps(json_data)
        data_bytes = json_str.encode('utf-8')
        
        conn.send(len(data_bytes).to_bytes(4, byteorder='big'))
        conn.send(data_bytes)
            
    
    def process_ik_request(self, data):
        try:
            if self.arm_ik is None:
                raise ValueError("IK solver not initialized")
                
            print("Processing IK request...")
            
            sol_q, sol_tauff = self.arm_ik.solve_ik(
                data['left_wrist'],
                data['right_wrist'], 
                data['current_lr_arm_q'],
                data['current_lr_arm_dq']
            )
            
            if sol_q is not None:
                print(f"IK solved successfully, sol_q shape: {sol_q.shape}")
            else:
                raise ValueError("IK solver returned None")
            
            return {
                'sol_q': sol_q,
                'success': True
            }
            
        except Exception as e:
            print(f"Error in IK calculation: {e}")
            return {
                'sol_q': data['current_lr_arm_q'], 
                'success': False,
                'error_msg': str(e)
            }

def main():
    server = IKServer()
    try:
        server.start_server()
    except Exception as e:
        print(f"Server error: {e}")

if __name__ == "__main__":
    main() 