"""CuTeDSL view over NCCL registered windows."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import ir

from ...resources import RegisteredWindowHandle
from . import _bindings as raw
from ._structs import _LLVMPtrType, ncclTeam as Team


class Window:
    """CuTeDSL view over an NCCL :c:type:`ncclWindow_t` handle.

    ``Window(resource)`` creates a host-mode JIT argument from a
    :py:class:`~nccl.core.RegisteredWindowHandle`. During tracing, CuTeDSL
    reconstructs the same class in value mode around an ``!llvm.ptr``.

    Args:
        resource: Registered window resource.

    Attributes:
        ptr: Device-side window handle, available while tracing.
    """

    def __init__(self, resource: RegisteredWindowHandle):
        if not isinstance(resource, RegisteredWindowHandle):
            raise TypeError(
                "Window expects an nccl.core.RegisteredWindowHandle, "
                f"got {type(resource).__name__}"
            )
        self._resource = resource
        self._ptr = None

    def __c_pointers__(self) -> list[int]:
        if self._resource is None:
            raise cutlass.DSLRuntimeError(
                "Window.__c_pointers__ is only available on a host-mode Window"
            )
        if not self._resource.is_valid:
            raise RuntimeError("RegisteredWindowHandle has been closed")
        return [self._resource._handle.address]

    @staticmethod
    def __get_mlir_types__() -> list[ir.Type]:
        return [_LLVMPtrType.mlir_type()]

    def __extract_mlir_values__(self) -> list[ir.Value]:
        return [self.ptr]

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> Window:
        obj = object.__new__(type(self))
        obj._resource = None
        obj._ptr = values[0]
        return obj

    @property
    def ptr(self) -> ir.Value:
        """Device-side :c:type:`ncclWindow_t` value."""
        if self._ptr is None:
            raise cutlass.DSLRuntimeError(
                "Window.ptr is only available while tracing CuTeDSL code"
            )
        return self._ptr

    def local_pointer(self, offset: int) -> ir.Value:
        """Translate ``offset`` to the local virtual address.

        Args:
            offset: Byte offset within the window.

        Returns:
            ``!llvm.ptr`` MLIR value.
        """
        return raw.ncclGetLocalPointer(self.ptr, offset)

    def lsa_pointer(self, offset: int, peer: int) -> ir.Value:
        """Translate ``offset`` to ``peer``'s LSA virtual address.

        Args:
            offset: Byte offset within the window.
            peer: LSA-team peer rank.

        Returns:
            ``!llvm.ptr`` MLIR value.
        """
        return raw.ncclGetLsaPointer(self.ptr, offset, peer)

    def peer_pointer(
        self, offset: int, peer: int, team: Team | None = None
    ) -> ir.Value:
        """Translate ``offset`` to ``peer``'s virtual address.

        Args:
            offset: Byte offset within the window.
            peer: Rank within ``team``.
            team: Team to address within. Defaults to ``None``.

        Returns:
            ``!llvm.ptr`` MLIR value.
        """
        if team is None:
            return raw.ncclGetPeerPointer(self.ptr, offset, peer)
        return raw.ncclGetPeerPointerTeam(self.ptr, offset, team, peer)

    def tensor(self, dtype, layout, offset: int = 0):
        """Construct a ``cute.Tensor`` view over the registered buffer.

        Canonical input to
        :py:meth:`~nccl.core.device.cute.Gin.put`: byte offset relative to
        the window and transfer size are derived from the tensor's iterator
        address and layout.

        Args:
            dtype: cutlass numeric type, such as ``cutlass.Int64``.
            layout: ``cute.Layout`` from ``cute.make_layout(...)``.
            offset: Byte offset within the window. Defaults to ``0``.

        Returns:
            ``cute.Tensor`` view at ``offset``.
        """
        return cute.make_tensor(
            cute.make_ptr(dtype, self.local_pointer(offset)),
            layout,
        )


@cutlass.register_jit_arg_adapter(RegisteredWindowHandle)
def _adapt_registered_window_handle(resource: RegisteredWindowHandle) -> Window:
    """Adapt a registered window resource to a CuTeDSL view."""
    return Window(resource)


__all__ = ["Window"]
