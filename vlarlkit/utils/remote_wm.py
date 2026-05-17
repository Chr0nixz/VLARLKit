"""
RemoteWM — a proxy that forwards world model calls to a WMServer via ZMQ.
"""

import logging
import pickle
from typing import Any, Optional, Sequence

import numpy as np
import zmq


_logger = logging.getLogger("vlarlkit.remote_wm")


class RemoteWM:
    """Remote proxy for a world model server."""

    def __init__(
        self,
        host: str,
        port: int,
        recv_timeout_ms: int = 300_000,
        max_retries: int = 1,
    ):
        """
        Args:
            host: World model server hostname or IP.
            port: World model server port.
            recv_timeout_ms: Timeout in milliseconds for receiving a response.
                             Defaults to 300 000 (5 minutes).
            max_retries: Max attempts per call before raising TimeoutError.
                         1 means no retry (fail immediately on first timeout).
        """
        self._host = host
        self._port = port
        self._recv_timeout_ms = recv_timeout_ms
        self._max_retries = max_retries
        self._closed = False
        self._ctx = zmq.Context()
        self._socket: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        """Create (or recreate) the REQ socket and connect to the server."""
        if self._socket is not None:
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.close()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self._recv_timeout_ms)
        self._socket.connect(f"tcp://{self._host}:{self._port}")

    def _send_and_recv(self, msg: dict, max_retries: int | None = None) -> Any:
        """Send a request and wait for the response, with timeout and retry."""
        retries = max_retries if max_retries is not None else self._max_retries
        method = msg.get("method", "unknown")

        for attempt in range(1, retries + 1):
            self._socket.send(pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL))
            try:
                response = pickle.loads(self._socket.recv())
            except zmq.Again:
                self._connect()
                if attempt < retries:
                    _logger.warning(
                        "RemoteWM '%s' timed out (attempt %d/%d, %ds). "
                        "Reconnected, retrying... [%s:%d]",
                        method,
                        attempt,
                        retries,
                        self._recv_timeout_ms // 1000,
                        self._host,
                        self._port,
                    )
                    continue
                raise TimeoutError(
                    f"RemoteWM: '{method}' call to {self._host}:{self._port} "
                    f"timed out after {retries} attempts "
                    f"({self._recv_timeout_ms / 1000:.0f}s each)"
                )
            if "error" in response:
                raise RuntimeError(f"World model server error: {response['error']}")
            return response["result"]

    def _call(self, method: str, **kwargs) -> Any:
        msg = {"method": method, "kwargs": kwargs}
        return self._send_and_recv(msg)

    @staticmethod
    def _prepare_observations(obs: dict) -> dict:
        """Normalize common keys while preserving extra world-model fields."""
        prepared = dict(obs)
        for key in ("main_images", "wrist_images", "extra_view_images", "states"):
            prepared.setdefault(key, None)
        prepared["task_descriptions"] = (
            list(obs["task_descriptions"])
            if "task_descriptions" in obs
            else None
        )
        return prepared

    def get_observations(
        self,
        observations: dict,
        actions,
        image_key: str = "main_images",
        **kwargs,
    ) -> dict:
        next_obs = self._call(
            "get_observations",
            observations=observations,
            actions=actions,
            image_key=image_key,
            **kwargs,
        )
        return self._prepare_observations(next_obs)

    def get_rewards(
        self,
        observations: dict,
        tasks: Optional[Sequence[str]] = None,
        image_key: str = "main_images",
        **kwargs,
    ) -> np.ndarray:
        return self._call(
            "get_rewards",
            observations=observations,
            tasks=tasks,
            image_key=image_key,
            **kwargs,
        )

    def step(
        self,
        observations: dict,
        actions,
        image_key: str = "main_images",
        wrist_image_key: str = "wrist_images",
        edit_kwargs: Optional[dict[str, Any]] = None,
        und_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        next_obs, rewards, terminations = self._call(
            "step",
            observations=observations,
            actions=actions,
            image_key=image_key,
            wrist_image_key=wrist_image_key,
            edit_kwargs=edit_kwargs,
            und_kwargs=und_kwargs,
        )
        return (
            self._prepare_observations(next_obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminations, dtype=bool),
        )

    def close(self):
        if self._closed:
            return
        try:
            self._send_and_recv({"method": "close", "kwargs": {}}, max_retries=1)
        finally:
            self._closed = True
            if self._socket is not None:
                self._socket.close()
            self._ctx.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
