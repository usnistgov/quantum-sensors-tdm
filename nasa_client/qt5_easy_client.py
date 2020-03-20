import asyncio
import time

import numpy as np
import zmq
from PyQt5.QtCore import QObject, QSocketNotifier, pyqtSignal

from . import easyClientNDFB


class AsyncEasyClient(QObject, easyClientNDFB.EasyClientNDFB):
    buffer_full = pyqtSignal()
    new_packet = pyqtSignal(dict, np.ndarray)

    def __init__(self):
        super(AsyncEasyClient, self).__init__()

        self.buffer = []
        self.buffer_len = 0
        self._max_len = 4 * 1024 * 1024  # 4 million bytes

        # This notifier acts like an asyncio reader.
        # But when  you use a quamash event loop, asyncio readers are implemented using
        # QSocketNotifier.
        self.notifier = QSocketNotifier(self.dataPort, QSocketNotifier.Read)
        self.notifier.activated.connect(self.collect)

        self._latest_packet_time_us = None

    def collect(self):
        """activated event handler of the underling zmq socket.
        It takes messages from the underlying zmq socket. Unless the buffer is full,
        it pushes the message into the buffer. Otherwise, it throws away the message."""
        try:
            while True:
                msg = self.dataPort.recv_multipart(zmq.NOBLOCK)[1]
                this_header = self.parse_packet_header(msg)
                header_size = this_header['header_bytes']
                self._latest_packet_time_us = (time.time() * 1e6, this_header['packet_timestamp'])

                data_array = self.packet_payload_to_data(msg[header_size:],
                                                         this_header['numpy_data_type'])
                self.new_packet.emit(this_header, data_array)

                # If buffer is unlimited or it's not full, store into the buffer.
                if self.max_len == -1 or self.buffer_len < self.max_len:
                    self.buffer_len += this_header['record_samples']
                    self.buffer.append((this_header, data_array))

                    # At the moment of full buffer, emits the buffer_full signal once.
                    if 0 <= self.max_len <= self.buffer_len:
                        self.buffer_full.emit()
        except zmq.Again:
            pass

    def clear(self):
        self.buffer.clear()
        self.buffer_len = 0

    @property
    def max_len(self):
        return self._max_len

    @max_len.setter
    def max_len(self, l):
        self._max_len = l
        while self.buffer_len >= l and not len(self.buffer):
            header, payload = self.buffer[0]
            self.buffer.pop(0)
            self.buffer_len -= header['record_samples']

    def wait_on_packet(self):
        """Returns a future which will be finished when a new packet arrives.
        """
        packet_arrived = asyncio.Future()

        def packet_listener(header, data):
            # new_packet signal means that new packet arrives.
            packet_arrived.set_result(header, data)
            self.new_packet.disconnect(packet_listener)

        self.new_packet.connect(packet_listener)

        return packet_arrived

    def wait_until(self, target_time):
        """Returns a future which will be done when a packet
        the server timestamp of which is later than the target time arrives.
        """
        time_passed = asyncio.Future()

        def packet_listener(header, _):
            if header['packet_timestamp'] > target_time:
                time_passed.set_result(None)
                self.new_packet.disconnect(packet_listener)

        self.new_pack.connect(packet_listener)

        return time_passed

    def wait_until_full(self):
        """Returns a future which will be done when the buffer gets full.
        """
        buffer_is_full = asyncio.Future()

        def on_buffer_full():
            buffer_is_full.set_result(None)
            self.buffer_full.disconnect(on_buffer_full)

        self.buffer_full.connect(on_buffer_full)
        return buffer_is_full

    async def get_new_data(self, delay_second=0.001, max_byte=4096):
        """"Collect new data after throwing away some packets for delay_second.
        """
        # Once new packet arrives, calculate the acquisition start time.
        await self.wait_on_packet()
        acquisition_time = self._latest_packet_time_us[1] + delay_second * 1e6
        self.max_len = 0

        await self.wait_until(acquisition_time)
        self.max_len = max_byte

        await self.wait_until_full()

        # Transforms the buffer into a right data type.
        return 0
