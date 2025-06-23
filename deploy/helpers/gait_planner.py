import numpy as np


class BipedalGaitPlanner:
    num_feet = 2

    def __init__(
        self,
        dt,
        frequencies=1.5,
        phase_offset=0.5,
        stance_ratio=0.6,
    ):
        self.dt = dt  # simulation duration of one frame
        self.frequencies = float(frequencies)  # the frequency of the gait
        self.phase_offset = float(phase_offset)  # the phase difference between the left and the right feet
        self.stance_ratio = float(stance_ratio)  # the proportion of the stance period in the whole gait period
        self.stance_middle_point = 0.3  # the initial state of the gait

        # gait states
        self.gait_index = self.stance_middle_point
        self.foot_indices = np.zeros(self.num_feet, dtype=np.float32)
        self.clock_inputs = np.zeros(self.num_feet, dtype=np.float32)

    def update_gait_phase(self, stop: bool = False):
        # update the gait phase
        self.gait_index = (self.gait_index + self.dt * self.frequencies) % 1.0
        self.foot_indices[0] = (self.gait_index + self.phase_offset) % 1.0
        self.foot_indices[1] = self.gait_index

        if stop:
            self.gait_index = self.stance_middle_point
            self.foot_indices[:] = self.stance_middle_point

        for i in range(self.num_feet):
            idx = self.foot_indices[i]
            if idx < self.stance_ratio:
                self.foot_indices[i] = 0.5 * idx / self.stance_ratio
            else:
                self.foot_indices[i] = 0.5 + 0.5 * (idx - self.stance_ratio) / (1 - self.stance_ratio)

            self.clock_inputs[i] = np.sin(2 * np.pi * self.foot_indices[i])
