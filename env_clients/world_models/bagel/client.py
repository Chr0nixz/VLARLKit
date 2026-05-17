from __future__ import annotations

import argparse
import io
import logging
import pickle
import signal
import sys
from typing import Any

import torch
import zmq

from .bagel_wm import BagelWM


logging.basicConfig(
    level=logging.INFO,
    format="[BagelWMServer %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger(__name__)


class BagelWMServer:
    """ZMQ REP server that hosts one BAGEL world model instance."""

    _CALLABLE_METHODS = frozenset({"get_observations", "get_rewards", "step", "close"})

    def __init__(self, host: str, port: int, world_model: BagelWM):
        self._world_model = world_model
        self._running = True
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REP)
        self._socket.bind(f"tcp://{host}:{port}")
        _logger.info("Listening on tcp://%s:%d", host, port)

    @staticmethod
    def _safe_pickle_loads(data: bytes) -> Any:
        """pickle.loads that remaps serialized CUDA tensors to a local device."""
        original = torch.storage._load_from_bytes
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

        def patched(buffer):
            return torch.load(io.BytesIO(buffer), map_location=map_location, weights_only=False)

        torch.storage._load_from_bytes = patched
        try:
            return pickle.loads(data)
        finally:
            torch.storage._load_from_bytes = original

    def serve_forever(self):
        _logger.info("Ready to serve requests.")
        try:
            while self._running:
                raw = self._socket.recv()
                msg = self._safe_pickle_loads(raw)
                response = self._dispatch(msg)
                self._socket.send(pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))
        except KeyboardInterrupt:
            _logger.info("Shutting down.")
        finally:
            self._socket.close()
            self._ctx.term()

    def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any]:
        method = msg.get("method")
        kwargs = msg.get("kwargs", {})

        if method not in self._CALLABLE_METHODS:
            return {"error": f"Method '{method}' is not allowed."}

        try:
            result = getattr(self._world_model, method)(**kwargs)
            if method == "close":
                self._running = False
            return {"result": result}
        except Exception as exc:
            _logger.exception("Error calling BagelWM.%s", method)
            return {"error": str(exc)}


def _build_world_model(args: argparse.Namespace) -> BagelWM:
    return BagelWM(
        load_model_path=args.load_model_path,
        gpu_id=args.gpu_id,
        max_mem_per_gpu=args.max_mem_per_gpu,
        offload_folder=args.offload_folder,
        edit_cfg_text_scale=args.edit_cfg_text_scale,
        edit_cfg_img_scale=args.edit_cfg_img_scale,
        edit_cfg_interval=(args.edit_cfg_interval_start, args.edit_cfg_interval_end),
        edit_timestep_shift=args.edit_timestep_shift,
        edit_num_timesteps=args.edit_num_timesteps,
        edit_cfg_renorm_min=args.edit_cfg_renorm_min,
        edit_cfg_renorm_type=args.edit_cfg_renorm_type,
        save_images=args.save_images,
        save_dir=args.save_dir,
        save_prefix=args.save_prefix,
        understand_max_tokens=args.understand_max_tokens,
        understand_do_sample=args.understand_do_sample,
        understand_temperature=args.understand_temperature,
    )


def main():
    parser = argparse.ArgumentParser(description="BAGEL World Model ZMQ Server")
    parser.add_argument("--load-model-path", type=str, required=True, help="Path to BAGEL model weights directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU id used by this server process")
    parser.add_argument("--max-mem-per-gpu", type=str, default="80GiB", help="Maximum memory per GPU")
    parser.add_argument(
        "--offload-folder",
        type=str,
        default=None,
        help="Folder used by accelerate when some BAGEL modules are offloaded to disk",
    )

    parser.add_argument("--edit-cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--edit-cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--edit-cfg-interval-start", type=float, default=0.4)
    parser.add_argument("--edit-cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--edit-timestep-shift", type=float, default=3.0)
    parser.add_argument("--edit-num-timesteps", type=int, default=50)
    parser.add_argument("--edit-cfg-renorm-min", type=float, default=0.0)
    parser.add_argument(
        "--edit-cfg-renorm-type",
        type=str,
        default="text_channel",
        choices=["global", "channel", "text_channel"],
    )
    parser.add_argument("--save-images", action="store_true", help="Save generated next images")
    parser.add_argument("--save-dir", type=str, default="./generated_images")
    parser.add_argument("--save-prefix", type=str, default="generated")

    parser.add_argument("--understand-max-tokens", type=int, default=1000)
    parser.add_argument("--understand-do-sample", action="store_true")
    parser.add_argument("--understand-temperature", type=float, default=0.3)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, lambda _signal, _frame: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda _signal, _frame: sys.exit(0))

    world_model = _build_world_model(args)
    server = BagelWMServer(host=args.host, port=args.port, world_model=world_model)
    server.serve_forever()


if __name__ == "__main__":
    main()
