"""Pythonic wrappers for ``ncclDevComm``, ``ncclWindow``, and ``ncclTeam``.

:class:`DevComm` is an explicit CuTeDSL view over a host-side
``DevCommResource``. Its JIT protocol passes a :py:class:`DevCommValue`
native struct by value and reconstructs the same public type in value mode
while tracing. Pointer storage is materialized lazily for device APIs whose
FFI ABI requires a pointer or reference. :class:`Window` remains a pointer
wrapper and exposes pointer translations (``win.local_pointer`` /
``lsa_pointer`` / ``peer_pointer``) plus :meth:`Window.tensor`, which
returns a ``cute.Tensor`` view over the registered buffer — the canonical
input to :meth:`Gin.put`.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl._mlir_helpers.op import dsl_user_op

from ...resources import DevCommResource
from . import _bindings as raw
from ._helpers import _to_ptr
from ._structs import (
    DevCommValue,
    _LLVMPtrType,
    ncclTeam as Team,
    ncclLsaBarrierHandle,
    ncclGinBarrierHandle,
    ncclMultimemHandle,
)
from .gin import Gin, _alloca_ncclGin_C
from .types import GinBackendMask


def _ptr_wrapper(cls):
    """Augment a ``@cute.native_struct``'s ``__init__`` for single-arg construction.

    A single positional integer or ptr-castable arg auto-converts to
    ``ptr=_to_ptr(x)``. Same-struct-value wrap and no-arg / kwarg modes
    still delegate to the native_struct-generated init.

    Args:
        cls: ``@cute.native_struct`` class to augment.

    Returns:
        ``cls`` (mutated in place).
    """
    native_init = cls.__init__

    @dsl_user_op
    def __init__(self, *args, loc=None, ip=None, **kwargs):
        if len(args) == 1 and not kwargs:
            x = args[0]
            is_struct_value = (
                isinstance(x, ir.Value) and x.type == type(self)._struct_type
            )
            if not is_struct_value:
                native_init(self, ptr=_to_ptr(x), loc=loc, ip=ip)
                return
        native_init(self, *args, loc=loc, ip=ip, **kwargs)

    cls.__init__ = __init__
    return cls


def _device_function_entry_block():
    """Return the enclosing device function's entry block."""
    op = ir.InsertionPoint.current.block.owner
    while op is not None:
        parent = op.parent
        if parent is not None and parent.name == "gpu.module":
            if len(op.regions) == 0 or len(op.regions[0].blocks) == 0:
                break
            return op.regions[0].blocks[0]
        op = parent
    raise cutlass.DSLRuntimeError(
        "Unable to find the device function entry block for DevComm.ptr"
    )


@dsl_user_op
def _materialize_dev_comm(value, *, loc=None, ip=None) -> ir.Value:
    """Materialize a by-value DevComm in the device function entry block."""
    entry_block = _device_function_entry_block()
    struct_value = value.__extract_mlir_values__()[0]
    with ir.InsertionPoint.at_block_begin(entry_block):
        ptr = llvm.alloca(
            res=ir.Type.parse("!llvm.ptr"),
            array_size=cutlass.Int32(1).ir_value(),
            elem_type=DevCommValue._struct_type,
            alignment=8,
            loc=loc,
        )
        llvm.store(struct_value, ptr, loc=loc)
    return ptr


class DevComm:
    """CuTeDSL view over a host-owned, by-value ``ncclDevComm``.

    ``DevComm(resource)`` creates a host-mode JIT argument that keeps the
    resource alive. CuTeDSL reconstructs value-mode instances without calling
    ``__init__``. The native struct is exposed as ``value`` for field reads
    while tracing; mutating it is unsupported. ``ptr`` is materialized once,
    on demand, and is never included in the JIT or kernel ABI.
    """

    def __init__(self, resource: DevCommResource):
        if not isinstance(resource, DevCommResource):
            raise TypeError(
                "DevComm expects an nccl.core.DevCommResource, "
                f"got {type(resource).__name__}"
        )
        self._resource = resource
        self.value = None
        self._materialized_ptr = None

    def __c_pointers__(self):
        if self._resource is None:
            raise cutlass.DSLRuntimeError(
                "DevComm.__c_pointers__ is only available on a host-mode "
                "DevComm"
            )
        return [self._resource.dev_comm.ptr]

    @staticmethod
    def __get_mlir_types__():
        return DevCommValue.__get_mlir_types__()

    def __extract_mlir_values__(self):
        return self.value.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values):
        obj = object.__new__(type(self))
        obj._resource = None
        obj.value = DevCommValue(values[0])
        obj._materialized_ptr = None
        return obj

    @property
    def ptr(self) -> ir.Value:
        """Device-local pointer to this value, materialized on first use."""
        if self._materialized_ptr is None:
            self._materialized_ptr = _materialize_dev_comm(self.value)
        return self._materialized_ptr

    # === Scalar fields ===

    @property
    def rank(self) -> cutlass.Int32:
        return self.value.rank

    @property
    def n_ranks(self) -> cutlass.Int32:
        return self.value.n_ranks

    @property
    def lsa_rank(self) -> cutlass.Int32:
        return self.value.lsa_rank

    @property
    def lsa_size(self) -> cutlass.Int32:
        return self.value.lsa_size

    # === Embedded barrier handles ===

    @property
    def lsa_barrier(self) -> ncclLsaBarrierHandle:
        return self.value.lsa_barrier

    @property
    def rail_gin_barrier(self) -> ncclGinBarrierHandle:
        return self.value.rail_gin_barrier

    @property
    def hybrid_lsa_barrier(self) -> ncclLsaBarrierHandle:
        return self.value.hybrid_lsa_barrier

    @property
    def hybrid_rail_gin_barrier(self) -> ncclGinBarrierHandle:
        return self.value.hybrid_rail_gin_barrier

    @property
    def world_gin_barrier(self) -> ncclGinBarrierHandle:
        return self.value.world_gin_barrier

    @property
    def lsa_multimem(self) -> ncclMultimemHandle:
        return self.value.lsa_multimem

    # === Team factories ===

    @property
    def team_world(self) -> Team:
        return raw.ncclTeamWorld(self.ptr)

    @property
    def team_lsa(self) -> Team:
        return raw.ncclTeamLsa(self.ptr)

    @property
    def team_rail(self) -> Team:
        return raw.ncclTeamRail(self.ptr)

    # === Gin factory ===

    def gin(self, backend: GinBackendMask, context_id: int) -> Gin:
        """Allocate and initialize a :class:`Gin` rooted on this comm.

        Args:
            backend: backend selection mask.
            context_id: GIN context id.

        Returns:
            Initialized :class:`Gin`.
        """
        storage = _alloca_ncclGin_C()
        raw.ncclGin_C_init(storage, backend, self, context_id)
        return Gin(ptr=storage)


@cutlass.register_jit_arg_adapter(DevCommResource)
def _adapt_dev_comm_resource(resource: DevCommResource) -> DevComm:
    """Adapt a device communicator resource to a CuTeDSL view."""
    return DevComm(resource)


@_ptr_wrapper
@cute.native_struct
class Window:
    """Pointer wrapper for ``ncclWindow_t``.

    Construct with ``Window(<integer handle from host>)``. Use
    :meth:`tensor` to obtain a ``cute.Tensor`` view over the registered
    buffer (the canonical input to :meth:`Gin.put`), or the
    ``local_pointer`` / ``lsa_pointer`` / ``peer_pointer`` methods to
    translate ``(offset, peer/team)`` tuples to raw virtual addresses.
    """

    ptr: _LLVMPtrType

    def local_pointer(self, offset: int) -> ir.Value:
        """Translate ``offset`` to the local virtual address.

        Args:
            offset: byte offset within the window.

        Returns:
            ``!llvm.ptr`` ir.Value.
        """
        return raw.ncclGetLocalPointer(self.ptr, offset)

    def lsa_pointer(self, offset: int, peer: int) -> ir.Value:
        """Translate ``offset`` to ``peer``'s LSA virtual address.

        Args:
            offset: byte offset within the window.
            peer: LSA-team peer rank.

        Returns:
            ``!llvm.ptr`` ir.Value.
        """
        return raw.ncclGetLsaPointer(self.ptr, offset, peer)

    def peer_pointer(self, offset: int, peer: int, team: Team = None) -> ir.Value:
        """Translate ``offset`` to ``peer``'s virtual address.

        Args:
            offset: byte offset within the window.
            peer: rank within ``team``.
            team: team to address within; default team if ``None``.

        Returns:
            ``!llvm.ptr`` ir.Value.
        """
        if team is None:
            return raw.ncclGetPeerPointer(self.ptr, offset, peer)
        return raw.ncclGetPeerPointerTeam(self.ptr, offset, team, peer)

    def tensor(self, dtype, layout, offset: int = 0):
        """Construct a ``cute.Tensor`` view over the registered buffer.

        Canonical input to :meth:`Gin.put`: byte offset (relative to the
        window) and transfer size are derived from the tensor's iterator
        address and layout.

        Args:
            dtype: cutlass numeric type (e.g. ``cutlass.Int64``).
            layout: ``cute.Layout`` from ``cute.make_layout(...)``.
            offset: byte offset within the window. Default 0.

        Returns:
            ``cute.Tensor`` view at ``offset``.
        """
        return cute.make_tensor(
            cute.make_ptr(dtype, self.local_pointer(offset)),
            layout,
        )


__all__ = [
    "Team",
    "DevComm",
    "DevCommValue",
    "Window",
]
