import time
import numpy as np
import torch
from typing import Union
from deploy.config import Config
import os
from os.path import join, isdir
import pickle
from copy import deepcopy
import threading
from deploy.helpers.policy_unified import SquatLowLevelPolicy, LocoLowLevelPolicy
from deploy.helpers.rotation_helper import get_gravity_orientation, transform_imu_data
from deploy.helpers.command_helper import create_damping_cmd, create_lower_damping_cmd, init_cmd_hg, init_cmd_go, MotorMode
from deploy.helpers.KF import ESEKF, IMUKF, IMUEKF
from deploy.controllers.handle_controller import UsbHandle

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

kf = IMUKF()
ekf = IMUEKF()
esekf = ESEKF(dt=1./50.)

torch.set_printoptions(precision=3)
np.set_printoptions(precision=3)

usb_left = UsbHandle("/dev/ttyACM0")
usb_right = UsbHandle("/dev/ttyACM1")
# Start receiving threads
usb_left.start_receiving()
usb_right.start_receiving()
# Register callbacks
usb_left.register_callback(usb_left.left_callback)
usb_right.register_callback(usb_right.right_callback)

class Controller:
    def __init__(self, config: Config, args) -> None:
        self.config = config

        # Initializing process variables
        self.qj = np.zeros(config.num_dof, dtype=np.float32)
        self.dqj = np.zeros(config.num_dof, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.obs = np.zeros(config.num_obs, dtype=np.float32)

        # Initializing process variables
        self.last_low_action = np.zeros(config.num_dof, dtype=np.float32)
        self.kps = config.kps
        self.kds = config.kds

        self.counter = 0
        self.transition_count = 0
    def set_transition_count(self):
        self.transition_count = self.config.transition_time

class Controller_loco(Controller):
    def __init__(self, config: Config, args) -> None:
        self.config = config
        super().__init__(config, args)
        self.loco_low_level_policy = LocoLowLevelPolicy(self.config)
        self.loco_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.stance_command = False

    # @ExecutionTime('compute_loco_cmd')
    def compute_loco_cmd(self, cmd_raw):
        if cmd_raw is None:
            return
        cmd = cmd_raw * self.config.max_cmd
        cmd_mask = np.abs(cmd) >= self.config.cmd_clip
        self.loco_cmd = cmd * cmd_mask
        # self.loco_cmd = cmd
        if self.stance_command:
            self.loco_cmd *= 0

    def run(self, cmd_raw, gravity_orientation, omega, qj_obs, dqj_obs, target_dof_pos):
        self.compute_loco_cmd(cmd_raw)
        # breakpoint()
        self.loco_low_level_policy.gait_planner.update_gait_phase(self.stance_command)
        self.obs, self.action, target_dof_pos[self.config.action_idx] = self.loco_low_level_policy.inference(
            self.loco_cmd,
            gravity_orientation,
            omega,
            qj_obs[self.config.dof_idx],
            dqj_obs[self.config.dof_idx])
        print('[run] loco_cmd', self.loco_cmd)
        return target_dof_pos


class Controller_squat(Controller):
    def __init__(self, config: Config, args) -> None:
        self.config = config
        super().__init__(config, args)
        self.squat_low_level_policy = SquatLowLevelPolicy(self.config)

        self.squat_cmd = np.array([0.75, 0.], dtype=np.float32)

    def compute_squat_cmd(self, cmd_raw):
        if cmd_raw is None:
            return
        # if self.squat_cmd[0] > 0.55 and cmd_raw[0] < 0:
        #     d_hp = cmd_raw * self.config.max_cmd / 800
        # else:
        d_hp = cmd_raw * self.config.max_cmd / 250
        self.squat_cmd += d_hp
        default_h = 0.75
        default_p = 0.0
        self.squat_cmd[0] = np.clip(self.squat_cmd[0], default_h - self.config.max_cmd[0], default_h)
        self.squat_cmd[1] = np.clip(self.squat_cmd[1], default_p, default_p + self.config.max_cmd[1])

    def run(self, cmd_raw, gravity_orientation, omega, qj_obs, dqj_obs, target_dof_pos):
        self.compute_squat_cmd(cmd_raw)
        self.obs, self.action, target_dof_pos[self.config.action_idx] = self.squat_low_level_policy.inference(
            self.squat_cmd,
            gravity_orientation,
            omega,
            qj_obs[self.config.dof_idx],
            dqj_obs[self.config.dof_idx])
        # print('[run] obs', self.obs)
        # print('[run] action', self.action)
        # print('[run] target_dof_pos', target_dof_pos)
        target_dof_pos[[5,11]] = 0.
        print('[run] squat_cmd', self.squat_cmd)
        return target_dof_pos

class Runner:
    def __init__(self, config: Config, args) -> None:
        self.config = config
        self.default_controller = Controller(config, args)
        if hasattr(config, "squat_config"):
            squat_config = Config(f"deploy/configs/{config.squat_config}")
            self.squat_controller = Controller_squat(squat_config, args)
        if hasattr(config, "loco_config"):
            loco_config = Config(f"deploy/configs/{config.loco_config}")
            self.loco_controller = Controller_loco(loco_config, args)
        self.counter = 0
        self.qj = np.zeros(config.num_dof, dtype=np.float32)
        self.dqj = np.zeros(config.num_dof, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.real_dof_idx = np.arange(config.num_dof)
        self.target_dof_pos = config.default_angles.copy()
        self.tau_record = np.zeros(config.num_dof, dtype=np.float32)

        self.transfer_to_loco = False
        self.transfer_to_squat = False
    def locoable(self):
        return self.squat_controller.squat_cmd[0] > 0.72 and self.squat_controller.squat_cmd[1] < 0.05

    def stopable(self):
        return abs(self.loco_controller.loco_cmd[0]) < 0.1  and self.loco_controller.loco_cmd[1] < 0.1 and self.loco_controller.loco_cmd[2] < 0.1

    def post_squat(self):
        if self.locoable():
            self.transfer_to_loco = usb_left.run_loco_signal
            if self.transfer_to_loco:
                self.loco_controller.stance_command = False
                self.loco_controller.set_transition_count()
                self.loco_controller.last_policy_target_dof_pos = self.target_dof_pos.copy()

    def post_loco(self):
        if self.stopable():
            if usb_left.stopgait_signal:
                self.loco_controller.stance_command = True
            self.transfer_to_squat = usb_right.run_squat_signal
            if self.transfer_to_squat:
                self.squat_controller.set_transition_count()

    def transition_loco(self):
        # print('[transition_loco]', self.loco_controller.transition_count)
        if self.loco_controller.transition_count > 0:
            gravity_orientation = get_gravity_orientation(self.quat)
            gravity_orientation = kf.update(gravity_orientation)
            cmd_raw = None
            target_dof_pos = self.target_dof_pos.copy()
            other_policy_target_dof_pos = self.squat_controller.run(
                cmd_raw, gravity_orientation,
                self.ang_vel, self.qj, self.dqj,
                target_dof_pos)
            alpha = self.loco_controller.transition_count / self.loco_controller.config.transition_time
            self.target_dof_pos = other_policy_target_dof_pos.copy()
            self.target_dof_pos[self.config.action_hl_idx] = (alpha * self.loco_controller.last_policy_target_dof_pos + (1 - alpha) * self.loco_controller.config.default_angles.copy())[self.config.action_hl_idx]
            self.loco_controller.transition_count -= 1
            return True
            # print('transition_loco')
        return False

    def transition_squat(self):
        # print('[transition_squat]', self.squat_controller.transition_count)
        if self.squat_controller.transition_count > 0:
            gravity_orientation = get_gravity_orientation(self.quat)
            gravity_orientation = kf.update(gravity_orientation)
            cmd_raw = None
            target_dof_pos = self.target_dof_pos.copy()
            other_policy_target_dof_pos = self.loco_controller.run(
                cmd_raw, gravity_orientation,
                self.ang_vel, self.qj, self.dqj,
                target_dof_pos)
            alpha = self.squat_controller.transition_count / self.squat_controller.config.transition_time
            self.target_dof_pos = alpha * other_policy_target_dof_pos + (1 - alpha) * self.target_dof_pos
            self.squat_controller.transition_count -= 1
            return True
            # print('transition_squat')
        return False

class Runner_online(Runner):
    def __init__(self, config: Config, args) -> None:
        self.config = config
        super().__init__(config, args)
        ChannelFactoryInitialize(0, args.net)

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            # self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            # self.lowcmd_publisher_.Init()

            # self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            # self.lowstate_subscriber.Init(self.LowStateHgHandler, 10) # self.LowStateHgHandler, 10

            self.low_cmd_state_thread = threading.Thread(target=self.LowStateHgHandler)
            self.low_cmd_state_thread.daemon = True
            self.low_cmd_state_thread.start()

        else:
            raise ValueError("Invalid msg_type")

        # wait for the proprioception subscriber to receive data
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

    def LowStateHgHandler(self): # , msg: LowStateHG
        # self.low_state = msg
        # self.low_state_timestamp = time.time()
        # self.mode_machine_ = self.low_state.mode_machine
        self.lowcmd_publisher_ = ChannelPublisher(self.config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber(self.config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init() # self.LowStateHgHandler, 10
        while True:
            msg = self.lowstate_subscriber.Read()
            self.low_state = msg
            self.low_state_timestamp = time.time()
            self.mode_machine_ = self.low_state.mode_machine
            time.sleep(0.001)
        # # pass


    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

class Runner_online_real(Runner_online):
    def __init__(self, config: Config, args) -> None:
        self.config = config
        super().__init__(config, args)
        self.save_data = args.save_data
        if self.save_data:
            self.save_data_dir = os.path.join(args.save_data_dir, config.exp_name)
            assert not self.save_data_dir.startswith("/")
            if isdir(self.save_data_dir):
                os.system("rm -r {}".format(self.save_data_dir))
            os.makedirs(self.save_data_dir, exist_ok=True)

            self.save_rawdata_thread = threading.Thread(target=self.save_rawdata)
            self.save_rawdata_thread.daemon = True
            self.save_rawdata_thread.start()

    def save_rawdata(self):
        while True:
            filepath = join(self.save_data_dir, "{}.pkl".format(str(self.counter).zfill(8)))
            # if filepath exist: return
            if os.path.exists(filepath) or self.counter < 1:
                continue
            rawdata = {
                "proprioception": {
                    "qj": self.qj,  # np.float32, shape = (num_actions,)
                    "dqj": self.dqj,  # np.float32, shape = (num_actions,)
                    "gravity_orientation": get_gravity_orientation(self.quat),  # np.float32, shape = (4,)
                    "ang_vel": self.ang_vel,  # np.float32, shape = (3,)
                },
                "squat_cmd": self.squat_controller.squat_cmd if hasattr(self.config, "squat_config") else -1,
                "loco_cmd": self.loco_controller.loco_cmd if hasattr(self.config, 'loco_config') else -1,  # np.float32, shape = (3,)
                "target_dof_pos": self.target_dof_pos,  # np.float32, shape = (num_actions,)
                "tau": self.tau_record,  # np.float32, shape = (num_actions,)
            }
            pickle.dump(rawdata, open(filepath, "wb"))
            time.sleep(0.02)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        if abs(self.tau_record).max() > 100: #
            print("large tau: ", np.arange(29)[abs(self.tau_record) > 100])
            create_damping_cmd(self.low_cmd)
            cmd = self.low_cmd
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def damping_state(self):
        print("Enter damping state.")
        print("Waiting for the Right_Start signal...")
        while not usb_right.start_signal:
            create_damping_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.real_dof_idx
        default_pos = self.config.default_angles.copy()
        dof_size = len(dof_idx)

        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            self.pd_control(self.default_controller, init_dof_pos * (1 - alpha) + default_pos * alpha)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

        print("Enter default pos state.")
        print("Waiting for the Right_Start_Long signal...")

        while not usb_right.run_signal:
            self.pd_control(self.default_controller, default_pos)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def pd_control(self, controller, target_pos):
        for j in range(len(self.real_dof_idx)):
            motor_idx = self.real_dof_idx[j]
            self.low_cmd.motor_cmd[motor_idx].q = target_pos[j]
            self.low_cmd.motor_cmd[motor_idx].dq = 0  # NOTE(axis
            self.low_cmd.motor_cmd[motor_idx].kp = controller.kps[j]
            self.low_cmd.motor_cmd[motor_idx].kd = controller.kds[j]
            self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.tau_record[motor_idx] = (target_pos[j] - self.low_state.motor_state[j].q) * controller.kps[j] - \
                                         self.low_state.motor_state[j].dq * controller.kds[j]


    # @ExecutionTime('refresh_prop')
    def refresh_prop(self):
        # Get the current joint position and velocity
        for i, j in enumerate(self.real_dof_idx):
            self.qj[i] = self.low_state.motor_state[j].q
            self.dqj[i] = self.low_state.motor_state[j].dq

        # imu_state quaternion: w, x, y, z
        self.quat = self.low_state.imu_state.quaternion
        self.ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)

        if self.config.imu_type == "torso":
            # h1 and h1_2 imu is on the torso
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            self.quat, self.ang_vel = transform_imu_data(waist_yaw=waist_yaw,
                                                         waist_yaw_omega=waist_yaw_omega,
                                                         imu_quat=self.quat,
                                                         imu_omega=self.ang_vel)  # NOTE(axis) only for h1 and h1_2 imu is on the torso ??

    def run_loco(self, debug=True, manual=True):
        self.refresh_prop()
        
        gravity_orientation = get_gravity_orientation(self.quat)
        gravity_orientation = kf.update(gravity_orientation)
        omega = self.ang_vel.copy()
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        target_dof_pos = self.target_dof_pos.copy()

        if manual:
            cmd_raw = self.loco_controller.config.cmd_debug.copy()
            cmd_raw[0] = usb_left.lx
            cmd_raw[1] = usb_left.ly
            cmd_raw[2] = usb_right.ry
        else:
            cmd_raw = None

        if not self.transition_loco():
            self.target_dof_pos = self.loco_controller.run(
                cmd_raw,
                gravity_orientation,
                omega,
                qj_obs,
                dqj_obs,
                target_dof_pos)

        self.pd_control(self.loco_controller, self.target_dof_pos)
        if usb_right.damping_signal:
            self.damping_state()
        # send the command
        if debug:
            create_damping_cmd(self.low_cmd)


        current_control_timestamp = time.time()
        time_until_next_step = self.config.control_dt - (current_control_timestamp - self.last_control_timestamp)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        current_control_timestamp = time.time()
        # print('loco fps: ', (1 / (current_control_timestamp - self.last_control_timestamp)), 'Hz')
        self.last_control_timestamp = current_control_timestamp

        self.send_cmd(self.low_cmd)

        self.counter += 1
        self.post_loco()

    def run_squat(self, debug=True, manual=True):
        self.refresh_prop()
        
        gravity_orientation = get_gravity_orientation(self.quat)
        gravity_orientation = kf.update(gravity_orientation)
        omega = self.ang_vel.copy()
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        target_dof_pos = self.target_dof_pos.copy()

        if manual:
            cmd_raw = self.squat_controller.config.cmd_debug.copy()
            cmd_raw[0] = usb_left.lx    # height
            cmd_raw[1] = usb_right.rx
        else:
            cmd_raw = None

        self.target_dof_pos = self.squat_controller.run(
            cmd_raw,
            gravity_orientation,
            omega,
            qj_obs,
            dqj_obs,
            target_dof_pos)

        self.transition_squat()
        self.pd_control(self.squat_controller, self.target_dof_pos)
        if usb_right.damping_signal:
            self.damping_state()
        # send the command
        if debug:
            create_damping_cmd(self.low_cmd)


        current_control_timestamp = time.time()
        time_until_next_step = self.config.control_dt - (current_control_timestamp - self.last_control_timestamp)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        current_control_timestamp = time.time()
        # print('squat fps: ', (1 / (current_control_timestamp - self.last_control_timestamp)), 'Hz')
        self.last_control_timestamp = current_control_timestamp

        self.send_cmd(self.low_cmd)
        self.counter += 1
        self.post_squat()

from enum import IntEnum
from multiprocessing import Process, shared_memory, Array, Lock
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_                               # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
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

unitree_tip_indices = [4, 9, 14] # [thumb, index, middle] in OpenXR
Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"
class Runner_online_real_dexhand(Runner_online_real):
    def __init__(self, config: Config, args) -> None:
        self.config = config
        super().__init__(config, args)

        # Shared Arrays for hand states
        self.left_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)
        self.left_q_target  = np.full(Dex3_Num_Motors, 0)
        self.right_q_target = np.full(Dex3_Num_Motors, 0)
        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.9
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
        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()
        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            print("[Dex3_1_Controller] Waiting to subscribe dds...")


    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        if abs(self.tau_record).max() > 100 or usb_right.damping_signal: # abs(self.tau_record).max() > 100 or
            print("large tau: ", np.arange(29)[abs(self.tau_record) > 100])
            create_damping_cmd(self.low_cmd)
            cmd = self.low_cmd
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

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

    def _subscribe_hand_state(self):
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()
        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()
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

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """set current left, right hand motor state target q"""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)

    def grasp(self):
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock=False)  # current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock=False)
        state_data = np.concatenate((np.array(self.left_hand_state_array[:]), np.array(self.right_hand_state_array[:])))

        if usb_left.left_hand_grasp_state == True:
            self.left_q_target = np.array([0.0, 0.5, 1.0, -1.0, -1.0, -1.0, -1.0])
        else:
            self.left_q_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        if usb_right.right_hand_grasp_state == True:
            self.right_q_target = np.array([0.0, -0.5, -1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            self.right_q_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


        action_data = np.concatenate((self.left_q_target, self.right_q_target))
        if dual_hand_state_array and dual_hand_action_array:
            with dual_hand_data_lock:
                dual_hand_state_array[:] = state_data
                dual_hand_action_array[:] = action_data

    def run_squat_hand(self, debug=True, manual=True):
        self.refresh_prop()
        # create observation
        gravity_orientation = get_gravity_orientation(self.quat)
        gravity_orientation = kf.update(gravity_orientation)
        omega = self.ang_vel.copy()
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        target_dof_pos = self.target_dof_pos.copy()

        if manual:
            cmd_raw = self.squat_controller.config.cmd_debug.copy()
            cmd_raw[0] = usb_left.lx
            cmd_raw[1] = usb_right.rx
        else:
            cmd_raw = None

        self.target_dof_pos = self.squat_controller.run(
            cmd_raw,
            gravity_orientation,
            omega,
            qj_obs,
            dqj_obs,
            target_dof_pos)

        self.transition_squat()
        self.pd_control(self.squat_controller, self.target_dof_pos)
        if usb_right.damping_signal:
            self.damping_state()
        # send the command
        if debug:
            create_damping_cmd(self.low_cmd)

        self.grasp()

        current_control_timestamp = time.time()
        time_until_next_step = self.config.control_dt - (current_control_timestamp - self.last_control_timestamp)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        current_control_timestamp = time.time()
        # print('squat hand fps: ', (1 / (current_control_timestamp - self.last_control_timestamp)), 'Hz')
        self.last_control_timestamp = current_control_timestamp


        self.ctrl_dual_hand(self.left_q_target, self.right_q_target)
        self.send_cmd(self.low_cmd)
        self.counter += 1
        self.post_squat()


import mujoco.viewer
import mujoco
class Runner_handle_mujoco(Runner):
    def __init__(self, config: Config, args) -> None:
        self.config = config
        super().__init__(config, args)

        # Load robot model
        self.m = mujoco.MjModel.from_xml_path(config.xml_path)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = self.config.simulation_dt
        self.last_control_timestamp = time.time()

        self.save_data = args.save_data
        if self.save_data:
            self.save_data_dir = os.path.join(args.save_data_dir, config.exp_name)
            assert not self.save_data_dir.startswith("/")
            if isdir(self.save_data_dir):
                os.system("rm -r {}".format(self.save_data_dir))
            os.makedirs(self.save_data_dir, exist_ok=True)
            self.save_rawdata_thread = threading.Thread(target=self.save_rawdata)
            self.save_rawdata_thread.daemon = True
            self.save_rawdata_thread.start()

    def save_rawdata(self):
        while True:
            filepath = join(self.save_data_dir, "{}.pkl".format(str(self.counter).zfill(8)))
            # if filepath exist: return
            if os.path.exists(filepath) or self.counter < 1:
                continue
            rawdata = {
                "proprioception": {
                    "qj": self.qj,  # np.float32, shape = (num_actions,)
                    "dqj": self.dqj,  # np.float32, shape = (num_actions,)
                    "gravity_orientation": get_gravity_orientation(self.quat),  # np.float32, shape = (4,)
                    "ang_vel": self.ang_vel,  # np.float32, shape = (3,)
                },
                "squat_cmd": self.squat_controller.squat_cmd if hasattr(self.config, "squat_config") else -1,
                "loco_cmd": self.loco_controller.loco_cmd if hasattr(self.config, 'loco_config') else -1,  # np.float32, shape = (3,)
                "target_dof_pos": self.target_dof_pos,  # np.float32, shape = (num_actions,)
                "tau": self.tau_record,  # np.float32, shape = (num_actions,)
            }
            pickle.dump(rawdata, open(filepath, "wb"))
            time.sleep(0.02)

    def pd_control(self, controller, target_q):
        """Calculates torques from position commands"""
        kp = controller.kps
        kd = controller.kds
        q = self.d.qpos[7:][self.real_dof_idx]
        target_dq = np.zeros_like(kd)
        dq = self.d.qvel[6:][self.real_dof_idx]
        return (target_q - q) * kp + (target_dq - dq) * kd

    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    def refresh_prop(self):
        self.qj = self.d.qpos[7:][self.config.dof_idx]
        self.dqj = self.d.qvel[6:][self.config.dof_idx]
        self.quat = self.d.qpos[3:7]
        self.ang_vel = self.d.qvel[3:6]

    def run_squat(self, manual=True):
        if self.counter % self.config.control_decimation == 0:
            # create observation
            self.refresh_prop()
            gravity_orientation = self.get_gravity_orientation(self.quat)
            target_dof_pos = self.target_dof_pos.copy()

            if manual:
                cmd_raw = self.squat_controller.config.cmd_debug.copy()
                cmd_raw[0] = usb_left.lx    # height
                cmd_raw[1] = usb_right.rx
                # print(cmd_raw)
            else:
                cmd_raw = None
            self.target_dof_pos = self.squat_controller.run(
                cmd_raw, gravity_orientation,
                self.ang_vel, self.qj, self.dqj,
                target_dof_pos)
            self.transition_squat()
        tau = self.pd_control(self.squat_controller, self.target_dof_pos)
        self.d.ctrl[:] = tau

        current_control_timestamp = time.time()
        # time.sleep(config.control_dt)
        time_until_next_step = self.config.simulation_dt - (current_control_timestamp - self.last_control_timestamp)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        current_control_timestamp = time.time()
        self.last_control_timestamp = current_control_timestamp

        mujoco.mj_step(self.m, self.d)
        self.counter += 1
        self.post_squat()

    def run_loco(self, manual=True):
        if self.counter % self.config.control_decimation == 0:
            # create observation
            self.refresh_prop()
            gravity_orientation = self.get_gravity_orientation(self.quat)
            target_dof_pos = self.target_dof_pos.copy()

            if manual:
                cmd_raw = self.loco_controller.config.cmd_debug.copy()
                cmd_raw[0] = usb_left.lx
                cmd_raw[1] = usb_left.ly
                cmd_raw[2] = usb_right.ry
                # print(cmd_raw)
            else:
                cmd_raw = None
            if not self.transition_loco():
                self.target_dof_pos = self.loco_controller.run(
                    cmd_raw, gravity_orientation,
                    self.ang_vel, self.qj, self.dqj,
                    target_dof_pos)

        tau = self.pd_control(self.loco_controller, self.target_dof_pos)
        # breakpoint()
        self.d.ctrl[:] = tau
        # print(tau)

        current_control_timestamp = time.time()
        # time.sleep(config.control_dt)
        time_until_next_step = self.config.simulation_dt - (current_control_timestamp - self.last_control_timestamp)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        current_control_timestamp = time.time()
        self.last_control_timestamp = current_control_timestamp

        mujoco.mj_step(self.m, self.d)
        self.counter += 1
        self.post_loco()

from mujoco import Renderer
class Runner_handle_mujoco_vision(Runner_handle_mujoco):
    def __init__(self, config: Config, args) -> None:
        self.config = config
        super().__init__(config, args)

        self.renderer = Renderer(self.m, width=640, height=480)
        left_camera_name = "left_eye"
        self.left_camera_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, left_camera_name)
        right_camera_name = "right_eye"
        self.right_camera_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, right_camera_name)

    def run_loco(self, manual=True):
        super().run_loco(manual)
        self.renderer.update_scene(self.d, camera=self.left_camera_id)
        left_image = self.renderer.render()
        self.renderer.update_scene(self.d, camera=self.right_camera_id)
        right_image = self.renderer.render()
        self.render_image = np.concatenate((left_image, right_image), axis=1)

    def run_squat(self, manual=True):
        super().run_squat(manual)
        self.renderer.update_scene(self.d, camera=self.left_camera_id)
        left_image = self.renderer.render()
        self.renderer.update_scene(self.d, camera=self.right_camera_id)
        right_image = self.renderer.render()
        self.render_image = np.concatenate((left_image, right_image), axis=1)

