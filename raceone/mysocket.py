from config import JUDGE_BOX_IP, JUDGE_BOX_PORT, TEAM_SHORT_NAME, RESULT_FOLDER
import socket
import struct
import binascii
import os
import time

class JudgeBoxClient:
    def __init__(self, ip=JUDGE_BOX_IP, port=JUDGE_BOX_PORT):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.isOpen = False
        self.connect()

    def connect(self):
        try:
            self.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.ip, self.port))
            self.isOpen = True
        except Exception as e:
            print(f"\033[31mERROR: Socket connect failed: {e}\033[0m")

    @staticmethod
    def print_hex(bytes_data):
        return binascii.hexlify(bytes_data).decode('utf-8')

    def send_result(self, savePath):
        if not self.isOpen:
            print(f"\033[31mERROR: Socket connect failed. File is saved to {savePath}\033[0m")
            return
        try:
            with open(savePath, 'rb') as f:
                Data = f.read()  # 保留原始换行
            # 保持本项目协议：结果 DataType = 1
            DataType = struct.pack('>I', 1)
            DataLength = struct.pack('>I', len(Data))
            message = DataType + DataLength + Data
            self.sock.send(message)
            print(f"[INFO] Sent result file: {savePath}")
        except Exception as e:
            print(f"\033[31m[ERROR] Failed to send result: {e}\033[0m")
        self.close()


    def send_start(self, round_num):
        if not self.isOpen:
            self.connect()
        try:
            content = f"{TEAM_SHORT_NAME}R1".encode()
            data_type = struct.pack('>I', 0)
            data_length = struct.pack('>I', len(content))
            message = data_type + data_length + content
            self.sock.send(message)
            print(f"[INFO] Sent start signal for round 1")
        except Exception as e:
            print(f"\033[31m[ERROR] Failed to send start signal: {e}\033[0m")

    
    def close(self):
        if getattr(self, 'isOpen', False):
            self.isOpen = False
            try:
                self.sock.close()
            except Exception:
                pass


if __name__ == '__main__':
    client = JudgeBoxClient(ip=JUDGE_BOX_IP, port=JUDGE_BOX_PORT)
    client.send_start(1)
    # time.sleep(1)
    # # 统一从配置的 RESULT_FOLDER 读取结果文件路径（跨平台）
    # result_path = os.path.join(RESULT_FOLDER, f"{TEAM_SHORT_NAME}-R1.txt")
    # client.send_result(result_path)
    # time.sleep(2)
