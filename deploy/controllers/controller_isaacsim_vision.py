import numpy as np
import time
import threading
from deploy.controllers.controller_isaacsim import Runner_handle_isaacsim
from deploy.config import Config
import cv2
import os
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from omni.isaac.core.utils.prims import create_prim
class Runner_handle_isaacsim_vision(Runner_handle_isaacsim):
    def __init__(self, config: Config, args) -> None:
        self.render_image = None
        self.camera_width = 640
        self.camera_height = 480
        self.frame_count = 0
        
        super().__init__(config, args)
        
        self._setup_cameras()
        
        
    def _setup_cameras(self):
        left_camera_prim_path = "/World/g1/torso_link/left_eye_camera"
        self.left_camera = Camera(
            prim_path=left_camera_prim_path,
            frequency=25,
            resolution=(self.camera_width, self.camera_height),
            translation = np.array([-0.82, 0.02, 0.45]),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 15, 0]), degrees=True)
        )
        
        right_camera_prim_path = "/World/g1/torso_link/right_eye_camera"
        self.right_camera = Camera(
            prim_path=right_camera_prim_path,
            frequency=25,
            resolution=(self.camera_width, self.camera_height),
            translation = np.array([-0.82, -0.02, 0.45]),
            orientation=rot_utils.euler_angles_to_quats(np.array([0, 15, 0]), degrees=True)
        )
        
    
    def initialize_cameras(self):
        """初始化摄像头"""
        try:
            if hasattr(self, 'left_camera'):
                self.left_camera.initialize()
                # self.left_camera.add_motion_vectors_to_frame()
                
            if hasattr(self, 'right_camera'):
                self.right_camera.initialize()
                # self.right_camera.add_motion_vectors_to_frame()
                
            
        except Exception as e:
            print(f"❌ Error initializing cameras: {e}")
    
    def update_render_image(self):
        try:
            if not hasattr(self, 'left_camera') or not hasattr(self, 'right_camera'):
                return
                
            left_rgba = self.left_camera.get_rgba()
            if left_rgba is not None:
                left_rgb = left_rgba[:, :, :3] 
            else:
                left_rgb = np.zeros((self.camera_height, self.camera_width, 3))
            
            right_rgba = self.right_camera.get_rgba()
            if right_rgba is not None:
                right_rgb = right_rgba[:, :, :3] 
            else:
                right_rgb = np.zeros((self.camera_height, self.camera_width, 3))
            
            self.render_image = np.hstack([left_rgb, right_rgb])
            
            self.frame_count += 1
            
        except Exception as e:
            print(f"⚠️ Error updating render image: {e}")
