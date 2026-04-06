"""TensorRT inference runtime. No PyTorch dependency."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _get_cudart():
    """Import cudart from whichever cuda-python version is installed."""
    try:
        # cuda-python >= 12.x new layout
        from cuda.bindings import runtime as cudart
        return cudart
    except ImportError:
        pass
    try:
        # cuda-python < 12.x legacy layout
        from cuda import cudart
        return cudart
    except ImportError:
        raise ImportError(
            "cuda-python is required. Install with: pip install cuda-python"
        )


class TrtModel:
    """Load and run a TensorRT engine.

    Usage::

        model = TrtModel("server_det.trt")
        output = model(pixel_values)  # numpy array
    """

    def __init__(self, engine_path: str | Path):
        import tensorrt as trt

        self._trt = trt
        self._cudart = _get_cudart()

        engine_path = Path(engine_path)
        logger.info("Loading TRT engine: %s", engine_path)

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)

        with open(engine_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()

        # Create CUDA stream
        cudart = self._cudart
        err, self._stream = cudart.cudaStreamCreate()
        assert err == 0 or str(err) == "cudaError_t.cudaSuccess", \
            f"cudaStreamCreate failed: {err}"

        # Inspect I/O tensors
        self._input_name = None
        self._output_name = None
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_name = name

        logger.info(
            "Engine loaded: input=%s, output=%s",
            self._input_name,
            self._output_name,
        )

    def __call__(self, pixel_values: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            pixel_values: Input tensor as numpy array (N,C,H,W), float32.

        Returns:
            Output tensor as numpy array.
        """
        cudart = self._cudart

        pixel_values = np.ascontiguousarray(pixel_values.astype(np.float32))

        # Set input shape (for dynamic shapes)
        self._context.set_input_shape(self._input_name, pixel_values.shape)

        # Infer output shape
        output_shape = tuple(
            self._context.get_tensor_shape(self._output_name)
        )
        output = np.empty(output_shape, dtype=np.float32)

        # Allocate device memory
        input_nbytes = pixel_values.nbytes
        output_nbytes = output.nbytes

        err, d_input = cudart.cudaMalloc(input_nbytes)
        assert err == 0 or "Success" in str(err), f"cudaMalloc failed: {err}"
        err, d_output = cudart.cudaMalloc(output_nbytes)
        assert err == 0 or "Success" in str(err), f"cudaMalloc failed: {err}"

        # Copy input H→D
        err, = cudart.cudaMemcpy(
            d_input, pixel_values.ctypes.data, input_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

        # Set tensor addresses
        self._context.set_tensor_address(self._input_name, int(d_input))
        self._context.set_tensor_address(self._output_name, int(d_output))

        # Execute
        self._context.execute_async_v3(stream_handle=int(self._stream))

        # Synchronize
        cudart.cudaStreamSynchronize(self._stream)

        # Copy output D→H
        cudart.cudaMemcpy(
            output.ctypes.data, d_output, output_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )

        # Free device memory
        cudart.cudaFree(d_input)
        cudart.cudaFree(d_output)

        return output

    def __del__(self):
        if hasattr(self, "_stream") and hasattr(self, "_cudart"):
            self._cudart.cudaStreamDestroy(self._stream)
