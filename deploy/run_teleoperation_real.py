# import onnxruntime as ort
from deploy.config import Config
from deploy.controllers.controller import Runner_online_real_dexhand, Runner_handle_mujoco  #, Runner_offline_mujoco

import time
import cv2
from multiprocessing import shared_memory, Process
import threading
from datetime import datetime
import os

from deploy.teleop.open_television.tv_wrapper import TeleVisionWrapper
from deploy.teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from deploy.teleop.image_server.image_client import ImageClient

import numpy as np
import torch

torch.set_printoptions(precision=3)
np.set_printoptions(precision=3)

tv_img_shape = (480, 1280, 3)
tv_img_dtype = np.uint8
tv_img_shm = shared_memory.SharedMemory(create=True, size=np.prod(tv_img_shape) * np.uint8().itemsize)
tv_img_array = np.ndarray(tv_img_shape, dtype=tv_img_dtype, buffer=tv_img_shm.buf)
img_client = ImageClient(tv_img_shape=tv_img_shape,
                         tv_img_shm_name=tv_img_shm.name)  # , server_address='10.100.6.192', port=8012

image_receive_thread = threading.Thread(target=img_client.receive_process, daemon=True)
image_receive_thread.start()

# television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
tv_wrapper = TeleVisionWrapper(True, tv_img_shape, tv_img_shm.name)

arm_ik = G1_29_ArmIK()


def get_output_dir():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join("record_real_images", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_images(shm_name, shape, dtype, interval=1.0 / 25.):
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


def deploy_real(args):
    # Load config
    config_path = f"deploy/configs/{args.config}"
    print(config_path)
    config = Config(config_path)

    runner = Runner_online_real_dexhand(config, args=args)
    # Enter the zero torque state, press the start key to continue executing
    runner.damping_state()
    # # Enter the default position state, press the A key to continue executing
    runner.move_to_default_pos()
    runner.last_control_timestamp = time.time()
    current_mode = "SQUAT"
    print('Squat mode!')
    print('Press Left_A to start the locomotion mode!')

    def tv_arms():
        head_rmat, left_wrist, right_wrist, left_hand, right_hand = tv_wrapper.get_data()
        current_lr_arm_q = runner.qj.copy()[15:29]
        current_lr_arm_dq = runner.dqj.copy()[15:29]
        # # solve ik using motor data and wrist pose, then use ik results to control arms.
        sol_q, sol_tauff = arm_ik.solve_ik(left_wrist, right_wrist, current_lr_arm_q, current_lr_arm_dq)
        sol_q = np.clip(sol_q, runner.target_dof_pos[runner.config.action_hl_idx] - 0.01,
                        runner.target_dof_pos[runner.config.action_hl_idx] + 0.01)
        runner.target_dof_pos[runner.config.action_hl_idx] = sol_q

    if args.save_image:
        p_record_video = Process(target=save_images, args=(tv_img_shm.name, tv_img_shape, tv_img_dtype))
        p_record_video.start()
    while True:
        if current_mode == "LOCOMOTION":
            runner.run_loco(debug=args.debug, manual=True)
            if runner.transfer_to_squat:
                current_mode = "SQUAT"
                print('Squat mode!')
                print('Press Left_A to start the locomotion mode!')

        elif current_mode == "SQUAT":
            tv_arms()
            runner.run_squat_hand(debug=args.debug, manual=True)
            if runner.transfer_to_loco:
                current_mode = "LOCOMOTION"
                print('Locomotion mode!')
                print('Press Right_A to start the squat mode!')

        # cv2.imshow("camera_view", cv2.cvtColor(runner.render_image, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)

    if args.save_image:
        p_record_video.terminate()
        p_record_video.join()
    tv_img_shm.unlink()
    tv_img_shm.close()
    print("Finally, exiting program...")
    exit(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, help="network interface")
    parser.add_argument("--config",
                        type=str,
                        help="config file name in the configs folder",
                        default="run_loco_squat_grasp.yaml")
    parser.add_argument("--save_data", action="store_true", help="whether saving the real data")
    parser.add_argument("--save_data_dir", type=str, help="where to save the data", default="./save_real_data")
    parser.add_argument("--save_image", action="store_true", help="whether saving the real image")
    parser.add_argument("--debug", action="store_true", help="")
    args = parser.parse_args()

    deploy_real(args)
