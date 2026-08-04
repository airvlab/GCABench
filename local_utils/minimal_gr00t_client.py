# gr00t_client_minimal.py
# Standalone PolicyClient — no gr00t install required.
# Reimplements gr00t/policy/server_client.py exactly, minus the server side.
# Only external deps: zmq, msgpack, numpy  (all already in your Isaac Lab env)

import io
from typing import Any

import msgpack
import numpy as np
import zmq


# ── Serializer (matches MsgSerializer in server_client.py exactly) ───────────

class _MsgSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=_MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=_MsgSerializer._decode)

    @staticmethod
    def _decode(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        # ModalityConfig objects can come back from get_modality_config;
        # return as plain dict — sufficient for eval use.
        return obj

    @staticmethod
    def _encode(obj):
        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
        raise TypeError(f"Unknown type: {type(obj)}")


# ── PolicyClient ──────────────────────────────────────────────────────────────

class PolicyClient:
    """
    Drop-in replacement for gr00t.policy.server_client.PolicyClient.

    Usage (same as the real class):
        policy = PolicyClient(host="localhost", port=5555)
        action_dict, info = policy.get_action(obs_dict)
        policy.reset()

    Start the GR00T server with:
        python scripts/inference_service.py \\
            --server \\
            --model-path <MODEL_PATH> \\
            --embodiment-tag <TAG> \\
            --port 5555
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ):
        self.host       = host
        self.port       = port
        self.timeout_ms = timeout_ms
        self.api_token  = api_token
        self._context   = zmq.Context()
        self._init_socket()

    def _init_socket(self):
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def _call(self, endpoint: str, data: dict | None = None, requires_input: bool = True) -> Any:
        """Send a request to the server and return the decoded response."""
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self._socket.send(_MsgSerializer.to_bytes(request))
            raw = self._socket.recv()
        except zmq.error.Again:
            # Timeout: REQ socket stuck — recreate before re-raising
            self._init_socket()
            raise TimeoutError(
                f"GR00T server at {self.host}:{self.port} did not respond "
                f"within {self.timeout_ms} ms."
            )

        if raw == b"ERROR":
            raise RuntimeError("GR00T server returned ERROR. Check the server log.")

        response = _MsgSerializer.from_bytes(raw)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    # ── Public API (matches gr00t PolicyClient exactly) ──────────────────────

    def get_action(
        self,
        observation: dict[str, Any],
        options:     dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Query the model for an action chunk.

        Args:
            observation: flat key/value obs dict (video.*, state.*, annotation.*)
            options:     optional extra kwargs forwarded to the policy

        Returns:
            action_dict  – e.g. {"action.x": (chunk,1), "action.gripper": (chunk,1), ...}
            info         – server metadata / timing dict (may be empty)
        """
        response = self._call(
            "get_action",
            {"observation": observation, "options": options},
        )
        # Server returns a list [action_dict, info_dict]; convert to tuple
        return tuple(response)

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy's internal state (e.g. diffusion history buffer)."""
        return self._call("reset", {"options": options})

    def ping(self) -> bool:
        """Return True if the server is reachable."""
        try:
            self._call("ping", requires_input=False)
            return True
        except (zmq.error.ZMQError, TimeoutError):
            return False

    def get_modality_config(self) -> dict:
        """Fetch the server's modality config (useful for debugging key names)."""
        return self._call("get_modality_config", requires_input=False)

    def close(self):
        self._socket.close()
        self._context.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass