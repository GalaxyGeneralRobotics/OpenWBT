import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_inv_np(q: np.ndarray) -> np.ndarray:
    """
    inverse of quaternions
    Accepts numpy.ndarray as input.
    """
    w = -1 * q[..., -1:]
    xyz = q[..., :3]
    return np.hstack([xyz, w])


def broadcast_quat_apply_np(q: np.ndarray, vec3: np.ndarray) -> np.ndarray:
    t = 2 * np.cross(q[..., :3], vec3, axis=-1)
    xyz = vec3 + q[..., 3, None] * t + np.cross(q[..., :3], t, axis=-1)
    return xyz


def broadcast_quat_multiply_np(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Multiply 2 quaternions. p.shape == q.shape
    """

    w: np.ndarray = p[..., 3:4] * q[..., 3:4] - np.sum(p[..., :3] * q[..., :3], axis=-1, keepdims=True)
    xyz: np.ndarray = (p[..., 3, None] * q[..., :3] + q[..., 3, None] * p[..., :3] +
                       np.cross(p[..., :3], q[..., :3], axis=-1))

    return np.concatenate([xyz, w], axis=-1)


def facing_to_world(pos_root, quat_root, point_facing):
    """
    convert the points from the facing coordinate system to the world coordinate system
    :param pos_root: the root position in the world coordinate system
    :param quat_root: the root orientation (quaternion, xyzw) in the world coordinate system
    :param point_facing: positions of the points in the facing coordinate system
    :return: positions of the points in the world coordinate system
    """
    R_root_to_world = R.from_quat(quat_root).as_matrix()

    x_axis_world = R_root_to_world[:, 0]
    y_axis_world = R_root_to_world[:, 1]

    x_axis_ground = np.array([x_axis_world[0], x_axis_world[1], 0.0])
    y_axis_ground = np.array([y_axis_world[0], y_axis_world[1], 0.0])

    x_axis_facing = x_axis_ground / np.linalg.norm(x_axis_ground)
    z_axis_facing = np.array([0.0, 0.0, 1.0])
    y_axis_facing = np.cross(z_axis_facing, x_axis_facing)

    R_facing_to_root = np.vstack([x_axis_facing, y_axis_facing, z_axis_facing]).T

    point_root = R_facing_to_root @ point_facing

    point_world = R_root_to_world @ point_root + pos_root

    return point_world


def get_euler_xyz(q):
    qx, qy, qz, qw = q[0], q[1], q[2], q[3]

    # roll
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = qw * qw - qx * qx - qy * qy + qz * qz
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * (np.pi / 2.0), np.arcsin(sinp))

    # yaw
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = qw * qw + qx * qx - qy * qy - qz * qz
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w
