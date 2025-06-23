# for dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_  # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
# for gripper
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_  # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
import time
import os
import sys
import threading
from multiprocessing import Process, shared_memory, Array, Lock

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from teleop.utils.weighted_moving_filter import WeightedMovingFilter
from teleop.handle_controller import UsbHandle

unitree_tip_indices = [4, 9, 14]  # [thumb, index, middle] in OpenXR
Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


class Dex3_1_Controller:

    def __init__(self,
                 left_hand_grasp_state,
                 right_hand_grasp_state,
                 dual_hand_state_array,
                 dual_hand_data_lock=None,
                 fps=100.0,
                 Unit_Test=False):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_hand_array: [input] Left hand skeleton data (required from XR device) to hand_ctrl.control_process

        right_hand_array: [input] Right hand skeleton data (required from XR device) to hand_ctrl.control_process

        dual_hand_data_lock: Data synchronization lock for dual_hand_state_array and dual_hand_action_array

        dual_hand_state_array: [output] Return left(7), right(7) hand motor state

        dual_hand_action_array: [output] Return left(7), right(7) hand motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing
        """
        print("Initialize Dex3_1_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)
        else:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3_Unit_Test)
            ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            print("[Dex3_1_Controller] Waiting to subscribe dds...")

        hand_control_process = Process(target=self.control_process,
                                       args=(left_hand_grasp_state, right_hand_grasp_state, dual_hand_state_array,
                                             dual_hand_data_lock))
        hand_control_process.daemon = True
        hand_control_process.start()

        print("Initialize Dex3_1_Controller OK!\n")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                # Update left hand state
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                # Update right hand state
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
            time.sleep(0.002)

    class _RIS_Mode:

        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """set current left, right hand motor state target q"""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)
        # print("hand ctrl publish ok.")

    def control_process(self,
                        left_hand_grasp_state,
                        right_hand_grasp_state,
                        dual_hand_state_array,
                        dual_hand_data_lock=None):
        self.running = True

        left_q_target = np.full(Dex3_Num_Motors, 0)
        right_q_target = np.full(Dex3_Num_Motors, 0)

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.2

        # initialize dex3-1's left hand cmd msg
        self.left_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id=id, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q = q
            self.left_msg.motor_cmd[id].dq = dq
            self.left_msg.motor_cmd[id].tau = tau
            self.left_msg.motor_cmd[id].kp = kp
            self.left_msg.motor_cmd[id].kd = kd

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id=id, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode
            self.right_msg.motor_cmd[id].q = q
            self.right_msg.motor_cmd[id].dq = dq
            self.right_msg.motor_cmd[id].tau = tau
            self.right_msg.motor_cmd[id].kp = kp
            self.right_msg.motor_cmd[id].kd = kd

        try:
            while self.running:
                start_time = time.time()

                if left_hand_grasp_state == True:
                    left_q_target = np.array([0.0, 0.0, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2])
                else:
                    left_q_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

                if right_hand_grasp_state == True:
                    right_q_target = np.array([0.0, 0.0, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2])
                else:
                    right_q_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

                # get dual hand action
                if dual_hand_state_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = np.concatenate((left_q_target, right_q_target))

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            print("Dex3_1_Controller has been closed.")


class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6


class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


if __name__ == "__main__":
    import argparse
    from teleop.open_television.tv_wrapper import TeleVisionWrapper
    from teleop.image_server.image_client import ImageClient

    parser = argparse.ArgumentParser()
    parser.add_argument('--dex', action='store_true', help='Use dex3-1 hand')
    parser.add_argument('--gripper', dest='dex', action='store_false', help='Use gripper')
    parser.set_defaults(dex=True)
    args = parser.parse_args()
    print(f"args:{args}\n")

    # # image
    # img_config = {
    #     'fps': 30,
    #     'head_camera_type': 'opencv',
    #     'head_camera_image_shape': [480, 1280],  # Head camera resolution
    #     'head_camera_id_numbers': [0],
    # }
    # ASPECT_RATIO_THRESHOLD = 2.0  # If the aspect ratio exceeds this value, it is considered binocular
    # if len(img_config['head_camera_id_numbers']) > 1 or (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
    #     BINOCULAR = True
    # else:
    #     BINOCULAR = False
    # # image
    # if BINOCULAR and not (img_config['head_camera_image_shape'][1] / img_config['head_camera_image_shape'][0] > ASPECT_RATIO_THRESHOLD):
    #     tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1] * 2, 3)
    # else:
    #     tv_img_shape = (img_config['head_camera_image_shape'][0], img_config['head_camera_image_shape'][1], 3)

    # img_shm = shared_memory.SharedMemory(create = True, size = np.prod(tv_img_shape) * np.uint8().itemsize)
    # img_array = np.ndarray(tv_img_shape, dtype = np.uint8, buffer = img_shm.buf)
    # img_client = ImageClient(tv_img_shape = tv_img_shape, tv_img_shm_name = img_shm.name)
    # image_receive_thread = threading.Thread(target = img_client.receive_process, daemon = True)
    # image_receive_thread.daemon = True
    # image_receive_thread.start()

    usb_left = UsbHandle("/dev/ttyACM0")
    usb_right = UsbHandle("/dev/ttyACM1")
    usb_left.start_receiving()
    usb_right.start_receiving()
    usb_left.register_callback(usb_left.left_callback)
    usb_right.register_callback(usb_right.right_callback)

    if args.dex:
        dual_hand_data_lock = Lock()
        left_hand_grasp_state = False
        right_hand_grasp_state = False
        dual_hand_state_array = Array('d', 14, lock=False)
        hand_ctrl = Dex3_1_Controller(left_hand_grasp_state, right_hand_grasp_state, dual_hand_state_array,
                                      dual_hand_data_lock)

    user_input = input("Please enter the start signal (enter 'r' to start the subsequent program):\n")
    if user_input.lower() == 'r':
        while True:
            left_hand_grasp_state = usb_left.left_hand_grasp_state
            right_hand_grasp_state = usb_right.right_hand_grasp_state

            # with dual_hand_data_lock:
            #     print(f"state : {list(dual_hand_state_array)} \naction: {list(dual_hand_action_array)} \n")
            time.sleep(0.01)
