import socket
import struct
import binascii
import time


class Socket:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = None
        self.isOpen = False
        self.connect()

    def connect(self):
        try:
            self.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3)
            self.sock.connect((self.ip, self.port))
            self.isOpen = True
            return True
        except Exception:
            self.isOpen = False
            return False

    @staticmethod
    def print_hex(bytes_data):
        return binascii.hexlify(bytes_data).decode("utf-8")

    def send_result(self, savePath):
        if not self.isOpen and not self.connect():
            print(
                "\033[31mERROR: Socket connect failed.File is saved to data/measure.txt\033[0m"
            )
            return False

        try:
            DataType = struct.pack(">I", 1)
            with open(savePath, "rb") as f:
                Data = f.read().decode().replace("\n", "").encode()
            DataLength = struct.pack(">I", len(Data))
            message = DataType + DataLength + Data
            if self.sock is None:
                return False
            self.sock.sendall(message)
            return True
        except Exception:
            return False
        finally:
            self.close()

    def send_start(self, roundNum):
        if not self.isOpen and not self.connect():
            print("\033[31mERROR: Socket connect failed and send start fail\033[0m")
            return False
        try:
            DataType = struct.pack(">I", 0)
            Data = "LingShui3D{}".format(roundNum).encode()
            DataLength = struct.pack(">I", len(Data))
            message = DataType + DataLength + Data
            if self.sock is None:
                return False
            self.sock.sendall(message)
            return True
        except Exception:
            return False

    def send_rotate(self):
        if not self.isOpen and not self.connect():
            print("\033[31mERROR: Socket connect failed.send rotate fail\033[0m")
            return False
        try:
            DataType = struct.pack(">I", 3)
            Data = "0000".encode()
            DataLength = struct.pack(">I", len(Data))
            message = DataType + DataLength + Data
            if self.sock is None:
                return False
            self.sock.sendall(message)
            return True
        except Exception:
            return False

    def close(self):
        self.isOpen = False
        if self.sock is None:
            return
        try:
            self.sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    ip = "127.0.0.1"
    port = 6666
    sock = Socket(ip, port)
    savePath = "data/measure.txt"
    import time

    sock.send_start(1)
    time.sleep(2)
    sock.send_result(savePath)

    time.sleep(5)

    sock.send_start(2)
    time.sleep(2)
    sock.send_result(savePath)
