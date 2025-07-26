
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless":False})
from deploy.config import Config
from deploy.controllers.controller_isaacsim import Runner_handle_isaacsim

import time
import cv2
from multiprocessing import shared_memory
import threading
from datetime import datetime
import os

from deploy.teleop.open_television.tv_wrapper import TeleVisionWrapper

import numpy as np
import torch
torch.set_printoptions(precision=3)
np.set_printoptions(precision=3)

def get_output_dir():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join("record_isaacsim_images", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_images(shm_name, shape, dtype, interval=1.0/25.):
    shm = shared_memory.SharedMemory(name=shm_name)
    img = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    output_dir = get_output_dir()
    counter = 0

    try:
        while True:
            img_copy = img.copy()
            filename = os.path.join(output_dir, f"{counter:04d}.png")
            cv2.imwrite(filename, img_copy)
            counter += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()

def deploy_handle_isaacsim(args):
    config_path = f"deploy/configs/{args.config}"
    config = Config(config_path)

    runner = Runner_handle_isaacsim(config, args=args)

    while simulation_app.is_running():
        runner.world.step(render=True)
    simulation_app.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='IsaacSim Teleoperation with Vision Support')
    parser.add_argument("--config", type=str, help="config file name in the configs folder", 
                       default="run_teleoperation.yaml")
    parser.add_argument("--save_image", action="store_true", help="whether saving the isaacsim image")
    args = parser.parse_args()

    deploy_handle_isaacsim(args)
