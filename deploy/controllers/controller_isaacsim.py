import time
import numpy as np
import torch
from deploy.config import Config
from deploy.controllers.controller import Runner, usb_left, usb_right

import omni.replicator.core as rep
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, Articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.stage import add_reference_to_stage
# from isaacsim.core.api.objects import VisualCuboid
# from isaacsim.sensors.camera import Camera
# import omni.isaac.core.utils.rotations as rot_utils
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdPhysics
class Runner_handle_isaacsim(Runner):
    def __init__(self, config: Config, args) -> None:
        self.config = config
        super().__init__(config, args)
        # global first_step
        # global reset_needed
        # self.first_step = True
        self.current_mode = "SQUAT" # NOTE: LOCOMOTION / SQUAT

        # Load robot model
        self.last_control_timestamp = time.time()
        print("IsaacSim dt: ", config.simulation_dt)

        self.world = World(
            physics_dt = config.simulation_dt, # 1/200 s
            rendering_dt = config.simulation_dt * config.control_decimation, #* config.control_decimation, # 1/200 s
            stage_units_in_meters = 1.0,
        )
        self.world.scene.add_default_ground_plane()
        rep.create.light(light_type="Dome", position=[0,0,0])

        prim_path = "/World/g1"
        usd_path = "resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.usd"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        self.g1 = SingleArticulation(
            prim_path=prim_path,
            name="g1",
            position=np.array([0, 0, 0.8]),
        )
        my_g1 = self.world.scene.add(self.g1)
        self.left_q_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.right_q_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        self.world.reset()
        self.world.add_physics_callback("physics_step", callback_fn=self.on_physics_step)
        self.initialize()

    #Juana: add here 
    def initialize(self):
        controller_dofs = [
            'left_hip_pitch', 'left_hip_roll', 'left_hip_yaw',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll', 
            'right_hip_pitch', 'right_hip_roll', 'right_hip_yaw',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll', 
            'waist_yaw', 'waist_roll', 'waist_pitch', 
            'left_shoulder_pitch', 'left_shoulder_roll', 'left_shoulder_yaw', 
            'left_elbow', 'left_wrist_roll', 'left_wrist_pitch', 'left_wrist_yaw', 
            'right_shoulder_pitch', 'right_shoulder_roll', 'right_shoulder_yaw', 
            'right_elbow', 'right_wrist_roll', 'right_wrist_pitch', 'right_wrist_yaw'
        ]

        usd_joint_names = self.g1.dof_names

        usd_joint_base_names = [name.replace('_joint', '') for name in usd_joint_names if 'hand' not in name]
        left_hand_joint_indices = [i for i, name in enumerate(usd_joint_names) if name.startswith('left_hand')]
        right_hand_joint_indices = [i for i, name in enumerate(usd_joint_names) if name.startswith('right_hand')]
        left_hand_usd_names_by_urdf = [
            "left_hand_thumb_0_joint",   # URDF idx 0
            "left_hand_thumb_1_joint",   # URDF idx 1
            "left_hand_thumb_2_joint",   # URDF idx 2
            "left_hand_middle_0_joint",  # URDF idx 3
            "left_hand_middle_1_joint",  # URDF idx 4
            "left_hand_index_0_joint",   # URDF idx 5
            "left_hand_index_1_joint",   # URDF idx 6
        ]

        right_hand_usd_names_by_urdf = [
            "right_hand_thumb_0_joint",   # URDF idx 0
            "right_hand_thumb_1_joint",   # URDF idx 1
            "right_hand_thumb_2_joint",   # URDF idx 2
            "right_hand_index_0_joint",   # URDF idx 3
            "right_hand_index_1_joint",   # URDF idx 4
            "right_hand_middle_0_joint",  # URDF idx 5
            "right_hand_middle_1_joint",  # URDF idx 6
        ]

        self.left_usd_to_urdf_idx = [
            usd_joint_names.index(name)
            for name in left_hand_usd_names_by_urdf
        ]

        self.right_usd_to_urdf_idx = [
            usd_joint_names.index(name)
            for name in right_hand_usd_names_by_urdf
        ]

        lower_body_keywords = ['hip', 'waist', 'knee', 'ankle', 'pelvis', 'torso']

        armatures = np.zeros(len(usd_joint_names), dtype=np.float32)

        for i, name in enumerate(usd_joint_names):
            if any(kw in name for kw in lower_body_keywords):
                armatures[i] = 0.01
            elif 'hand' in name:
                armatures[i] = 0.001
            else:
                armatures[i] = 0.001

        self.urdf_to_usd = [-1] * len(usd_joint_base_names)
        self.usd_to_urdf = []
        for i, name in enumerate(controller_dofs):
            idx = usd_joint_base_names.index(name)
            self.usd_to_urdf.append(idx)
            self.urdf_to_usd[idx] = i

        self.g1.initialize()

        body_names = self.g1._articulation_view.body_names

        for link_name in body_names:
            if 'wrist' not in link_name:
                continue
            link_path = f'/World/g1/{link_name}'

            prim = get_prim_at_path(link_path)

            mass_api = UsdPhysics.MassAPI.Apply(prim)
            current_mass = mass_api.GetMassAttr().Get()

            new_mass = current_mass * 0.5
            mass_api.GetMassAttr().Set(new_mass)
            print(f"Set mass of {link_name} from {current_mass} to {new_mass}")

        self.g1._articulation_view.set_armatures(armatures)
        self.g1.set_solver_position_iteration_count(4)
        self.g1.set_solver_velocity_iteration_count(0)
        self.g1.set_enabled_self_collisions(False)
        kps = np.concatenate([self.squat_controller.kps[self.urdf_to_usd], np.array(self.config.hand_kps * 2)])
        kds = np.concatenate([self.squat_controller.kds[self.urdf_to_usd], np.array(self.config.hand_kds * 2)])
        self.g1._articulation_view.set_gains(kps, kds)
        print("sim dt: ", self.world.get_physics_dt())
        print("rendering dt: ", self.world.get_rendering_dt())
        print("joints: ", self.g1.dof_names)
        print("armatures: ", self.g1._articulation_view.get_armatures())
        print("position iteration count: ", self.g1.get_solver_position_iteration_count())
        print("velocity iteration count: ", self.g1.get_solver_velocity_iteration_count())
        stiff, dampings = self.g1._articulation_view.get_gains()
        print("stiff: ", stiff)
        print("damping: ", dampings)
        print("max effort: ", self.g1._articulation_view.get_max_efforts())

    def pd_control(self, controller, target_q):
        """Calculates torques from position commands"""
        kp = controller.kps
        kd = controller.kds
        q = self.qj[self.real_dof_idx]
        target_dq = np.zeros_like(kd)
        dq = self.dqj[self.real_dof_idx]
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
        self.qj = self.g1.get_joint_positions()[self.usd_to_urdf][self.config.dof_idx]
        self.dqj = self.g1.get_joint_velocities()[self.usd_to_urdf][self.config.dof_idx]
        position, orientation = self.g1.get_world_pose()
        angular_velocity = self.g1.get_angular_velocity()
        self.quat = orientation
        self.ang_vel = angular_velocity

    def on_physics_step(self, step_size) -> None:
        print(self.transfer_to_loco, self.transfer_to_squat, self.current_mode)
        if self.transfer_to_squat:
            self.transfer_to_squat = False
            if self.current_mode == "LOCOMOTION":
                self.current_mode = "SQUAT"
                kps = np.concatenate([self.squat_controller.kps[self.urdf_to_usd], np.array(self.config.hand_kps * 2)])
                kds = np.concatenate([self.squat_controller.kds[self.urdf_to_usd], np.array(self.config.hand_kds * 2)])
                self.g1._articulation_view.set_gains(kps, kds)
        elif self.transfer_to_loco:
            self.transfer_to_loco = False
            if self.current_mode == "SQUAT":
                self.current_mode = "LOCOMOTION"
                kps = np.concatenate([self.loco_controller.kps[self.urdf_to_usd], np.array(self.config.hand_kps * 2)])
                kds = np.concatenate([self.loco_controller.kds[self.urdf_to_usd], np.array(self.config.hand_kds * 2)])
                self.g1._articulation_view.set_gains(kps, kds)
        if self.current_mode == "LOCOMOTION":
            self.run_loco()
        elif self.current_mode == "SQUAT":
            self.run_hand()
            self.run_squat()

    def run_hand(self):
        if usb_left.left_hand_grasp_state == True:
            self.left_q_target = np.array([0.0, 0.5, 1.0, -1.0, -1.0, -1.0, -1.0])
        else:
            self.left_q_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        if usb_right.right_hand_grasp_state == True:
            self.right_q_target = np.array([0.0, -0.5, -1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            self.right_q_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


    def run_squat(self, manual=True):
        if self.counter % self.config.control_decimation == 0:
            self.refresh_prop()
            gravity_orientation = self.get_gravity_orientation(self.quat)
            target_dof_pos = self.target_dof_pos.copy()

            if manual:
                cmd_raw = self.squat_controller.config.cmd_debug.copy()
                cmd_raw[0] = usb_left.lx
                cmd_raw[1] = usb_right.rx
            else:
                cmd_raw = None
            self.target_dof_pos = self.squat_controller.run(
                cmd_raw, gravity_orientation,
                self.ang_vel, self.qj, self.dqj,
                target_dof_pos)
            self.transition_squat()
        full_target_dof_pos = np.zeros(43)
        full_target_dof_pos[:29] = self.target_dof_pos[self.urdf_to_usd]
        full_target_dof_pos[np.array(self.left_usd_to_urdf_idx, dtype=np.int32)] = self.left_q_target
        full_target_dof_pos[np.array(self.right_usd_to_urdf_idx, dtype=np.int32)] = self.right_q_target
        self.g1.apply_action(ArticulationAction(joint_positions=full_target_dof_pos))

        real_time = True
        if real_time:
            current_control_timestamp = time.time()
            time_until_next_step = self.config.simulation_dt - (current_control_timestamp - self.last_control_timestamp)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            current_control_timestamp = time.time()
            self.last_control_timestamp = current_control_timestamp
        self.counter += 1
        self.post_squat()

    def run_loco(self, manual=True):
        if self.counter % self.config.control_decimation == 0:
            self.refresh_prop()
            gravity_orientation = self.get_gravity_orientation(self.quat)
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
                    cmd_raw, gravity_orientation,
                    self.ang_vel, self.qj, self.dqj,
                    target_dof_pos)
        self.g1.apply_action(ArticulationAction(joint_positions=self.target_dof_pos[self.urdf_to_usd], joint_indices = np.arange(len(self.urdf_to_usd), dtype=np.int32)))

        real_time = True
        if real_time:
            current_control_timestamp = time.time()
            time_until_next_step = self.config.simulation_dt - (current_control_timestamp - self.last_control_timestamp)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            current_control_timestamp = time.time()
            self.last_control_timestamp = current_control_timestamp
        self.counter += 1
        self.post_loco()