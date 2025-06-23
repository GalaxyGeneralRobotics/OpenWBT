import numpy as np
import numpy.linalg as la
import deploy.helpers.transformations as tr
import math


class IMUKF:

    def __init__(self, process_noise=1e-4, measurement_noise=1e-2):
        self.x = None
        self.P = np.eye(3)
        self.Q = np.eye(3) * process_noise
        self.R = np.eye(3) * measurement_noise
        self.H = np.eye(3)
        self.I = np.eye(3)

    def update(self, z):
        """
        z: currently observed gravity direction
        """
        z = np.array(z)

        if self.x is None:
            self.x = z.copy()
            return self.x

        # prediction step
        x_pred = self.x
        P_pred = self.P + self.Q

        # update step
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y
        self.P = (self.I - K @ self.H) @ P_pred

        return self.x


class IMUEKF:

    def __init__(self, process_noise=1e-4, measurement_noise=1e-3):
        self.x = np.zeros(6)
        self.P = np.eye(6) * 1.0
        self.Q = np.eye(6) * process_noise
        self.R = np.eye(6) * measurement_noise
        self.H = np.eye(6)

    def update(self, gravity_from_quat, ang_vel_measurement):
        z = np.concatenate([gravity_from_quat, ang_vel_measurement])

        # Prediction step
        x_pred = self.x
        P_pred = self.P + self.Q

        # Measurement update
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y
        self.P = (np.eye(6) - K @ self.H) @ P_pred

        # 返回平滑后的重力方向和角速度
        # return smoothened gravity direction and angular velocity
        return self.x[:3], self.x[3:]


class ImuParameters:

    def __init__(self):
        self.sigma_w_n = 0.01  # gyro noise.  rad/sqrt(s)
        self.sigma_w_b = 0.0001  # gyro bias


class ESEKF:

    def __init__(self, dt):
        """
        :param init_nominal_state: [q, b_g] (4+3=7D)
        """
        self.nominal_state = None
        self.dt = dt

        self.imu_parameters = ImuParameters()

        # noise covariance matrix (6x6): [gyro_noise, gyro_bias_noise]
        noise_covar = np.zeros((6, 6))
        noise_covar[0:3, 0:3] = (self.imu_parameters.sigma_w_n**2) * np.eye(3)
        noise_covar[3:6, 3:6] = (self.imu_parameters.sigma_w_b**2) * np.eye(3)

        # G matrix: how noise affects the state (6x6 here)
        G = np.zeros((6, 6))
        G[0:3, 0:3] = -np.eye(3)  # gyro noise
        G[3:6, 3:6] = np.eye(3)  # gyro bias noise

        self.noise_covar = G @ noise_covar @ G.T  # process noise (6x6)
        self.error_covar = 0.01 * np.eye(6)  # initial error covariance

    def update(self, gt_measurement: np.array, imu_measurement: np.array):
        """
        :param imu_measurement: [t, w_m, a_m]
        :return:
        """
        measurement_covar = np.eye(3) * 0.015 * 0.015
        if self.nominal_state is None:
            self.nominal_state = np.zeros(7)
            self.nominal_state[:4] = gt_measurement
        # we predict error_covar first, because __predict_nominal_state will change the nominal state.
        self.__predict_error_covar(imu_measurement)
        self.__predict_nominal_state(imu_measurement)

        # H projects error_state (6D) to delta (6D)
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)  # rotation error
        # H[3:6, 3:6] = np.eye(3)  # gyro bias error

        PHt = self.error_covar @ H.T
        K = PHt @ la.inv(H @ PHt + measurement_covar)

        # measurement residual (delta)
        q_nom = self.nominal_state[0:4]
        if gt_measurement[0] < 0:
            gt_measurement *= -1
        delta_q = tr.quaternion_multiply(tr.quaternion_conjugate(q_nom), gt_measurement)
        if delta_q[0] < 0:
            delta_q *= -1

        angle = math.asin(la.norm(delta_q[1:4]))
        axis = np.zeros(3,) if math.isclose(angle, 0) else delta_q[1:4] / la.norm(delta_q[1:4])
        delta = np.zeros((3, 1))
        delta[0:3, 0] = angle * axis  # rotation difference

        # update state estimate
        errors = K @ delta
        dq = tr.quaternion_about_axis(la.norm(errors[0:3, 0]), errors[0:3, 0])
        self.nominal_state[0:4] = tr.quaternion_multiply(q_nom, dq)
        self.nominal_state[0:4] /= la.norm(self.nominal_state[0:4])

        self.nominal_state[4:7] += errors[3:6, 0]  # update gyro bias

        # update error covariance
        self.error_covar = (np.eye(6) - K @ H) @ self.error_covar
        self.error_covar = 0.5 * (self.error_covar + self.error_covar.T)  # symmetrize

        # apply G matrix for quaternion error injection
        G = np.eye(6)
        G[0:3, 0:3] = np.eye(3) - tr.skew_matrix(0.5 * errors[0:3, 0])
        self.error_covar = G @ self.error_covar @ G.T
        return self.nominal_state[0:4]

    def __predict_nominal_state(self, imu_measurement: np.array):
        """
        Predict step for simplified ESEKF using only gyro.
        :param imu_measurement: np.array([dt, wx, wy, wz])
        """
        q = self.nominal_state[0:4]  # quaternion
        w_b = self.nominal_state[4:7]  # gyro bias

        dt = self.dt
        w_m = imu_measurement

        w = w_m - w_b

        angle = la.norm(w)
        if angle < 1e-6:
            return

        axis = w / angle
        R_w = tr.rotation_matrix(dt * angle, axis)
        dq = tr.quaternion_from_matrix(R_w, True)
        q_next = tr.quaternion_multiply(q, dq)

        if q_next[0] < 0:
            q_next *= -1

        self.nominal_state[0:4] = q_next

    def __predict_error_covar(self, imu_measurement: np.array):
        w_m = imu_measurement  # shape=(3,)
        w_b = self.nominal_state[4:7]  # shape=(3,)
        w = w_m - w_b

        q = self.nominal_state[0:4]

        F = np.zeros((6, 6))

        F[0:3, 0:3] = -tr.skew_matrix(w)
        F[0:3, 3:6] = -np.eye(3)

        dt = self.dt

        Fdt = F * dt
        Fdt2 = Fdt @ Fdt
        Fdt3 = Fdt2 @ Fdt
        Phi = np.eye(6) + Fdt + 0.5 * Fdt2 + (1. / 6.) * Fdt3

        Qc_dt = 0.5 * dt * self.noise_covar

        self.error_covar = Phi @ (self.error_covar + Qc_dt) @ Phi.T + Qc_dt
