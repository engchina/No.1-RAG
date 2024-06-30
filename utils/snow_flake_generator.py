import threading
import time


class SnowflakeGenerator:
    def __init__(self, datacenter_id, worker_id):
        self.datacenter_id = datacenter_id
        self.worker_id = worker_id
        self.sequence = 0
        self.last_timestamp = -1

        # Bit lengths for different parts
        self.datacenter_id_bits = 5
        self.worker_id_bits = 5
        self.sequence_bits = 12

        # Maximum values
        self.max_datacenter_id = -1 ^ (-1 << self.datacenter_id_bits)
        self.max_worker_id = -1 ^ (-1 << self.worker_id_bits)
        self.max_sequence = -1 ^ (-1 << self.sequence_bits)

        # Shift amounts
        self.worker_id_shift = self.sequence_bits
        self.datacenter_id_shift = self.sequence_bits + self.worker_id_bits
        self.timestamp_shift = self.sequence_bits + self.worker_id_bits + self.datacenter_id_bits

        self.lock = threading.Lock()

    def _current_milliseconds(self):
        return int(time.time() * 1000)

    def _til_next_millis(self, last_timestamp):
        timestamp = self._current_milliseconds()
        while timestamp <= last_timestamp:
            timestamp = self._current_milliseconds()
        return timestamp

    def generate_id(self):
        with self.lock:
            timestamp = self._current_milliseconds()

            if timestamp < self.last_timestamp:
                raise ValueError("Clock moved backwards. Refusing to generate id.")

            if timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & self.max_sequence
                if self.sequence == 0:
                    timestamp = self._til_next_millis(self.last_timestamp)
            else:
                self.sequence = 0

            self.last_timestamp = timestamp

            return ((timestamp - 1288834974657) << self.timestamp_shift) | \
                (self.datacenter_id << self.datacenter_id_shift) | \
                (self.worker_id << self.worker_id_shift) | \
                self.sequence


def generate_unique_id(prefix: str, datacenter_id: int, worker_id: int) -> str:
    generator = SnowflakeGenerator(datacenter_id, worker_id)
    snowflake_id = generator.generate_id()
    return f"{prefix}{snowflake_id}"
