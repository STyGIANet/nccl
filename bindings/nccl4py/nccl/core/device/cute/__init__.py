"""User-facing CuTeDSL bindings for the NCCL device API.

Typical usage::

    import cutlass
    import cutlass.cute as cute
    import nccl.core.device.cute as nccl_cute

    @cute.kernel
    def my_kernel(dev_comm: nccl_cute.DevComm, win: cutlass.Int64):
        win = nccl_cute.Window(win)
        team = dev_comm.team_world
        coop = nccl_cute.cta()
        gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)

        src = win.tensor(cutlass.Int64, cute.make_layout(1))
        dst = win.tensor(cutlass.Int64, cute.make_layout(1))
        src[0] = 1234
        gin.put(team, peer, win, dst, win, src, coop,
                is_signal=True, signal_id=1)
        ...

    @cute.jit
    def launch(dev_comm: nccl_cute.DevComm, win: cutlass.Int64):
        my_kernel(dev_comm, win).launch(grid=[1, 1, 1], block=[32, 1, 1])

    resource = nccl_comm.create_dev_comm(requirements=reqs)
    dev_comm = nccl_cute.DevComm(resource)
    launch(dev_comm, win.handle)
"""

try:
    import cutlass.cute  # noqa: F401
except ImportError as e:
    raise ImportError(
        "nccl.core.device.cute requires the nvidia-cutlass-dsl package. "
        "Install it via the matching extra:\n"
        "    pip install 'nccl4py[cu12]'   # for CUDA 12\n"
        "    pip install 'nccl4py[cu13]'   # for CUDA 13"
    ) from e

from . import types, coop, comm, gin, barrier
from .types import *    # MemoryOrder, GinFenceLevel, GinBackendMask
from .coop import *     # Coop, cta, warp, thread, lanes, warp_span
from .comm import *     # Team, DevComm, DevCommValue, Window
from .gin import *      # Gin
from .barrier import *  # session classes + factories
from ._helpers import device_bitcode_path

__all__ = [
    "types",
    "coop",
    "comm",
    "gin",
    "barrier",
    *types.__all__,
    *coop.__all__,
    *comm.__all__,
    *gin.__all__,
    *barrier.__all__,
    "device_bitcode_path",
]
