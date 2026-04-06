"""TensorRT inference runtime. No PyTorch dependency."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class TrtModel:
    """Load and run a TensorRT engine.

    Usage::

        model = TrtModel("server_det.trt")
        output = model(pixel_values)  # numpy array
    """

    def __init__(self, engine_path: str | Path):
        import tensorrt as trt
        from cuda import cuda, cudart

        self._trt = trt
        self._cuda = cuda
        self._cudart = cudart

        engine_path = Path(engine_path)
        logger.info("Loading TRT engine: %s", engine_path)

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)

        with open(engine_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()

        # Create CUDA stream
        err, self._stream = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess, f"cudaStreamCreate failed: {err}"

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
        trt = self._trt

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

        err, d_input = cudart.cudaMallocAsync(input_nbytes, self._stream)
        assert err == cudart.cudaError_t.cudaSuccess
        err, d_output = cudart.cudaMallocAsync(output_nbytes, self._stream)
        assert err == cudart.cudaError_t.cudaSuccess

        # Copy input H→D
        err, = cudart.cudaMemcpyAsync(
            d_input, pixel_values.ctypes.data, input_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self._stream,
        )
        assert err == cudart.cudaError_t.cudaSuccess

        # Set tensor addresses
        self._context.set_tensor_address(self._input_name, int(d_input))
        self._context.set_tensor_address(self._output_name, int(d_output))

        # Execute
        self._context.execute_async_v3(stream_handle=int(self._stream))

        # Copy output D→H
        err, = cudart.cudaMemcpyAsync(
            output.ctypes.data, d_output, output_nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self._stream,
        )
        assert err == cudart.cudaError_t.cudaSuccess

        # Synchronize
        err, = cudart.cudaStreamSynchronize(self._stream)
        assert err == cudart.cudaError_t.cudaSuccess

        # Free device memory
        cudart.cudaFreeAsync(d_input, self._stream)
        cudart.cudaFreeAsync(d_output, self._stream)

        return output

    def __del__(self):
        if hasattr(self, "_stream"):
            self._cudart.cudaStreamDestroy(self._stream)
