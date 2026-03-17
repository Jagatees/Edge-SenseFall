import smbus
import time


class MMWave:
    def __init__(self, bus=1, address=0x52):
        self.bus = smbus.SMBus(bus)
        self.address = address
        self.prev_distance = None

    def read_raw(self):
        data = self.bus.read_i2c_block_data(self.address, 0x00, 16)
        return data

    def parse(self, raw):
        distance = None
        velocity = None
        presence = 0

        try:
            parts = raw.split()
            for p in parts:
                if "distance" in p:
                    distance = float(p.split(":")[1])
                if "velocity" in p:
                    velocity = float(p.split(":")[1])
                if "presence" in p:
                    presence = int(p.split(":")[1])
        except:
            pass

        return distance, velocity, presence

    def detect_human(self):
        raw = self.read_raw()
        if not raw:
            return False, 0, 0

        distance, velocity, presence = self.parse(raw)

        return presence == 1, distance, velocity

    def detect_fall_pattern(self, distance, velocity):
        if self.prev_distance is None:
            self.prev_distance = distance
            return False

        height_drop = self.prev_distance - distance
        self.prev_distance = distance

        if velocity and height_drop:
            if velocity > 2.0 and height_drop > 0.5:
                return True

        return False
