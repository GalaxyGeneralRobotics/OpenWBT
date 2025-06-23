import serial
import threading
import time
from typing import Callable, Dict, Optional


class UsbHandle:
    KEY_1 = 1
    KEY_2 = 2
    KEY_3 = 3
    KEY_4 = 4
    KEY_5 = 5
    KEY_6 = 6
    KEY_PULLEY = 7
    KEY_7 = 8
    KEY_8 = 9
    KEY_9 = 10

    KEY_STATUS = ["KEY_DOWN", "KEY_UP", "KEY_LONG", "KEY_CLICK"]
    KEY_NAMES = ["KEY_1", "KEY_2", "KEY_3", "KEY_4", "KEY_5", "KEY_6", "KEY_PULLEY", "KEY_7", "KEY_8", "KEY_9"]

    def __init__(self, port: str):
        self.serial = serial.Serial()
        self.serial.port = port
        self.serial.baudrate = 921600
        self.serial.timeout = 0
        self.running = False
        self.callback: Optional[Callable[[Dict], None]] = None
        self.buffer = bytearray()
        self.thread: Optional[threading.Thread] = None

        self.start_signal = False
        self.run_signal = False
        self.run_loco_signal = False
        self.run_squat_signal = False
        self.damping_signal = False
        self.stopgait_signal = False
        self.left_hand_grasp_state = True
        self.right_hand_grasp_state = True
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0

        try:
            self.serial.open()
        except serial.SerialException as e:
            print(f"Failed to open {port}: {e}")
            raise

    def start_receiving(self):
        if not self.serial.is_open:
            return
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop_receiving(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        self.serial.close()

    def register_callback(self, callback: Callable[[Dict], None]):
        self.callback = callback

    def _receive_loop(self):
        while self.running:
            # Read all available data
            data = self.serial.read(self.serial.in_waiting or 1)
            if data:
                self.buffer.extend(data)
                self._process_buffer()

            time.sleep(0.001)

    def _process_buffer(self):
        while len(self.buffer) >= 10:
            # Find frame start (0xDE 0xED)
            start = -1
            for i in range(len(self.buffer) - 9):
                if self.buffer[i] == 0xDE and self.buffer[i + 1] == 0xED:
                    # Check frame end (0xEA 0xAE)
                    if self.buffer[i + 8] == 0xEA and self.buffer[i + 9] == 0xAE:
                        start = i
                        break
                    else:
                        continue

            if start == -1:
                # Remove invalid data (keep last 9 bytes)
                if len(self.buffer) > 9:
                    self.buffer = self.buffer[-9:]
                return

            # Extract and parse frame
            frame = self.buffer[start:start + 10]
            key_frame = {
                'keyIdx': frame[2],
                'value1': int.from_bytes(frame[4:6], byteorder='big', signed=True),
                'value2': int.from_bytes(frame[6:8], byteorder='big', signed=True)
            }
            # print(f"Received frame: {key_frame}")

            # Trigger callback
            if self.callback:
                self.callback(key_frame)

            # Remove processed data
            self.buffer = self.buffer[start + 10:]

    def left_callback(self, frame: Dict):
        key_idx = frame['keyIdx']
        status = frame['value1'] - 1

        if key_idx != self.KEY_PULLEY:
            idx = key_idx - 1
            # print(f"left {self.KEY_NAMES[idx]} -> {self.KEY_STATUS[status]}")
            # L1左手抓取，L2左手放开
            if self.KEY_NAMES[idx] == "KEY_9" and self.KEY_STATUS[status] == "KEY_DOWN":
                self.left_hand_grasp_state = True
                # print('Left hand grasp: ', self.left_hand_grasp_state)
            elif self.KEY_NAMES[idx] == "KEY_8" and self.KEY_STATUS[status] == "KEY_DOWN":
                self.left_hand_grasp_state = False
                # print('Left hand release: ', self.left_hand_grasp_state)

            # 左A切换到loco mode
            elif self.KEY_NAMES[idx] == "KEY_1" and self.KEY_STATUS[status] == "KEY_DOWN":
                self.run_loco_signal = True
                # print('Run loco signal: ', self.run_loco_signal)
            elif self.KEY_NAMES[idx] == "KEY_1" and self.KEY_STATUS[status] == "KEY_UP":
                self.run_loco_signal = False
                # print('Run loco signal: ', self.run_loco_signal)

            elif self.KEY_NAMES[idx] == "KEY_2" and self.KEY_STATUS[status] == "KEY_DOWN":
                self.stopgait_signal = True
            elif self.KEY_NAMES[idx] == "KEY_2" and self.KEY_STATUS[status] == "KEY_UP":
                self.stopgait_signal = False
        else:
            self.lx = (1853 - frame['value2']) / 2500
            self.ly = (frame['value1'] - 1853) / 2500
            # print(f"left_pulley | value1={frame['value1']} value2={frame['value2']}")

    def right_callback(self, frame: Dict):
        key_idx = frame['keyIdx']
        status = frame['value1'] - 1

        if key_idx != self.KEY_PULLEY:
            idx = key_idx - 1
            # print(f"right {self.KEY_NAMES[idx]} -> {self.KEY_STATUS[status]}")
            # 短按右start初始模式，长按右start run policy
            if self.KEY_NAMES[idx] == "KEY_5" and self.KEY_STATUS[status] == "KEY_DOWN":
                self.start_signal = True
                # print('Start signal: ', self.start_signal)
            elif self.KEY_NAMES[idx] == "KEY_5" and self.KEY_STATUS[status] == "KEY_LONG":
                self.run_signal = True
                # print('Default pos signal: ', self.run_signal)
            elif self.KEY_NAMES[idx] == "KEY_5" and self.KEY_STATUS[status] == "KEY_UP":
                self.start_signal = False
                self.run_signal = False

            # 短按右D阻尼模式，长按右D退出程序
            elif self.KEY_NAMES[idx] == "KEY_2" and self.KEY_STATUS[status] == "KEY_DOWN":
                self.damping_signal = True
                # print('Zero torque signal: ', self.damping_signal)
            elif self.KEY_NAMES[idx] == "KEY_2" and self.KEY_STATUS[status] == "KEY_LONG":
                self.damping_signal = True
                # print('Exit signal: ', self.exit_signal)
            # elif self.KEY_NAMES[idx] == "KEY_2" and self.KEY_STATUS[status] == "KEY_UP":
            #     self.damping_signal = False
            # self.exit_signal = False
            # print('Zero torque signal: ', self.damping_signal)

            # 右A切换到squat mode
            elif self.KEY_NAMES[idx] == "KEY_1" and self.KEY_STATUS[status] == "KEY_DOWN":
                self.run_squat_signal = True
                # print('Run squat signal: ', self.run_squat_signal)
            elif self.KEY_NAMES[idx] == "KEY_1" and self.KEY_STATUS[status] == "KEY_UP":
                self.run_squat_signal = False
                # print('Run squat signal: ', self.run_squat_signal)

            # R1右手抓取，R2右手放开
            elif self.KEY_NAMES[idx] == "KEY_9" and self.KEY_STATUS[status] == "KEY_DOWN":
                self.right_hand_grasp_state = True
                # print('Right hand grasp: ', self.right_hand_grasp_state)
            elif self.KEY_NAMES[idx] == "KEY_8" and self.KEY_STATUS[status] == "KEY_DOWN":
                self.right_hand_grasp_state = False
                # print('Right hand release: ', self.right_hand_grasp_state)
        else:
            self.rx = (1853 - frame['value2']) / 2500
            self.ry = (frame['value1'] - 1853) / 2500


# def main():
#     # Initialize USB devices
#         try:
#             usb_left = UsbHandle("/dev/ttyACM2")
#             usb_right = UsbHandle("/dev/ttyACM3")
#         except:
#             return

#         # Start receiving threads
#         usb_left.start_receiving()
#         usb_right.start_receiving()

#         # # Register callbacks
#         usb_left.register_callback(usb_left.left_callback)
#         usb_right.register_callback(usb_right.right_callback)

#         # Keep main thread alive
#         try:
#             while True:
#                 time.sleep(0.5)
#                 # print(usb_left.run_loco_signal)
#                 # print('lin_vel_x = ', usb_left.lx)
#                 # print('lin_vel_y = ', usb_left.ly)
#         except KeyboardInterrupt:
#             usb_left.stop_receiving()
#             usb_right.stop_receiving()

# if __name__ == "__main__":
#     main()
