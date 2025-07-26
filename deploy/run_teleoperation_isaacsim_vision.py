
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
from deploy.config import Config
from deploy.controllers.controller_isaacsim_vision import Runner_handle_isaacsim_vision

import time
import cv2
from multiprocessing import shared_memory, Process
import threading
from datetime import datetime
import os

from deploy.teleop.open_television.tv_wrapper import TeleVisionWrapper
from deploy.helpers.ik_client import IKClient

ik_client = IKClient()

import numpy as np
import torch
torch.set_printoptions(precision=3)
np.set_printoptions(precision=3)

tv_img_shape = (480, 1280, 3)
tv_img_dtype = np.uint8
tv_img_shm = shared_memory.SharedMemory(create=True, size=int(np.prod(tv_img_shape) * np.uint8().itemsize))
tv_img_array = np.ndarray(tv_img_shape, dtype=tv_img_dtype, buffer=tv_img_shm.buf)

tv_wrapper = TeleVisionWrapper(True, tv_img_shape, tv_img_shm.name)

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

def deploy_handle_isaacsim_vision(args):
    config_path = f"deploy/configs/{args.config}"
    config = Config(config_path)

    runner = Runner_handle_isaacsim_vision(config, args=args)
    
    runner.initialize_cameras()

    def tv_arms():
        try:
            head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()
            current_lr_arm_q = runner.qj.copy()[15:29]
            current_lr_arm_dq = runner.dqj.copy()[15:29]
            
            sol_q = ik_client.solve_ik_remote(left_wrist, right_wrist, current_lr_arm_q, current_lr_arm_dq)
            
            sol_q = np.clip(sol_q, runner.target_dof_pos[15:29] - 0.05,
                            runner.target_dof_pos[15:29] + 0.05)
            runner.target_dof_pos[15:29] = sol_q
                
        except Exception as e:
            print(f"❌ TV arms control error: {e}")

    if args.save_image:
        p_record_video = Process(target=save_images, args=(tv_img_shm.name, tv_img_shape, tv_img_dtype))
        p_record_video.start()

    while simulation_app.is_running():
        runner.world.step(render=True)
        if runner.current_mode == 'SQUAT' and runner.counter % 10 == 0:
            tv_arms()
        runner.update_render_image()
        try:
            np.copyto(tv_img_array, np.array(runner.render_image))
        except Exception as e:
            print(f"❌ Error saving image: {e}")

    ik_client.disconnect()
    simulation_app.close()


    if args.save_image:
        p_record_video.terminate()
        p_record_video.join()
    tv_img_shm.unlink()
    tv_img_shm.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='IsaacSim Teleoperation with Vision Support')
    parser.add_argument("--config", type=str, help="config file name in the configs folder", 
                       default="run_teleoperation.yaml")
    parser.add_argument("--save_image", action="store_true", help="whether saving the isaacsim image")
    args = parser.parse_args()

    
    deploy_handle_isaacsim_vision(args) 