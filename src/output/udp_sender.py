import socket
from typing import Optional

from src.exceptions import OutputError
from src.utils.logger import get_logger


class UDPSender:
    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        self._host = host
        self._port = port
        self._socket: Optional[socket.socket] = None
        self._logger = get_logger(__name__)

    def open(self) -> None:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._logger.info(f"UDP sender initialized: {self._host}:{self._port}")
        except OSError as e:
            raise OutputError(f"Failed to create UDP socket: {e}")

    def send(self, data: str) -> None:
        if self._socket is None:
            raise OutputError("Socket not initialized. Call open() first.")

        try:
            self._socket.sendto(
                data.encode("utf-8"),
                (self._host, self._port)
            )
        except OSError as e:
            self._logger.error(f"Failed to send UDP packet: {e}")

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
            self._logger.info("UDP socket closed")

    @property
    def is_opened(self) -> bool:
        return self._socket is not None
