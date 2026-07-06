"""1 MiB NCCL GIN put example using CuTeDSL ``cute.Tensor`` views.

Demonstrates the canonical workflow for the nccl4py CuTeDSL device API:
register two windows (send / recv) on the host, construct ``cute.Tensor``
views over them inside the kernel via Window.tensor, issue a
single Gin.put with a completion signal, wait on the signal on
the destination rank, and validate the payload host-side.

Run with two MPI ranks::

    mpirun -n 2 python main.py
"""

import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py required. Install with: pip install mpi4py")
    sys.exit(1)

try:
    from cuda.core import Device, system
except ImportError:
    print("ERROR: cuda.core required. Install with: pip install cuda-core")
    sys.exit(1)

try:
    import cupy as cp
except ImportError:
    print("ERROR: cupy required. Install with: pip install cupy-cuda13x (or cupy-cuda12x)")
    sys.exit(1)

import cutlass
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE
import nccl.core as nccl
import nccl.core.device.cute as nccl_cute

# 1 MiB transfer: 131072 Int64 elements * 8 bytes = 1,048,576 bytes.
NUM_ELEMS = 1024 * 1024 // 8
DST_RANK = 1
SIGNAL_ID = 1

@cute.kernel
def test_nccl_put_kernel(
    dev_comm: nccl_cute.DevComm,
    send_win: nccl_cute.Window,
    recv_win: nccl_cute.Window,
):
    """Issue a 1 MiB GIN put from rank 0 to rank 1 via ``cute.Tensor`` views.

    Runs with exactly 2 ranks. Two separate windows make the data flow
    explicit:

        * ``send_win`` is rank 0's source buffer (pre-filled with
          ``arange`` on the host).
        * ``recv_win`` is rank 1's destination buffer (validated host-side
          after sync).

    NCCL window registrations are collective, so both ranks register both
    windows; rank 0's ``recv_win`` and rank 1's ``send_win`` exist but go
    unused.

    Args:
        dev_comm: Value-mode nccl_cute.DevComm reconstructed while
            tracing from the host-mode instance.
        send_win: CuTeDSL view of the registered source window.
        recv_win: CuTeDSL view of the registered destination window.
    """
    tidx, _, _ = cute.arch.thread_idx()
    team = dev_comm.team_world
    gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)
    coop = nccl_cute.cta()

    # cute.Tensor views spanning the full 1 MiB of each window.
    send = send_win.tensor(cutlass.Int64, cute.make_layout(NUM_ELEMS))
    recv = recv_win.tensor(cutlass.Int64, cute.make_layout(NUM_ELEMS))

    if team.nRanks >= 2:
        if 0 == team.rank:
            if 0 == tidx:
                cute.printf(f"Before Put: send[0]={send[0]} send[{NUM_ELEMS - 1}]={send[NUM_ELEMS - 1]}\n")
            gin.put(
                team,
                DST_RANK,
                recv_win, recv,   # destination window + tensor (lives on the peer)
                send_win, send,   # source window + tensor (local)
                coop,
                is_signal=True,
                signal_id=SIGNAL_ID,
                signal_op=0,
                signal_op_arg=1,
            )
        if 1 == team.rank:
            gin.wait_signal(coop, signal=SIGNAL_ID, least=1)
            if 0 == tidx:
                cute.printf(f"After Put:  recv[0]={recv[0]} recv[{NUM_ELEMS - 1}]={recv[NUM_ELEMS - 1]}\n")


@cute.jit
def test_nccl_put(
        #dev_comm: nccl.DevCommResource,
        #send_win: nccl.RegisteredWindowHandle,
        #recv_win: nccl.RegisteredWindowHandle
        dev_comm: nccl_cute.DevComm,
        send_win: nccl_cute.Window,
        recv_win: nccl_cute.Window
    ):
    """Launch test_nccl_put_kernel with a single-warp grid.

    A @cute.jit function can take these arguments in two host-side
    forms. As of cutlass-dsl 4.5, CuTeDSL validates each parameter
    annotation against the argument BEFORE the JIT arg adapter runs, so
    the annotation determines which form the caller may pass:

    1. Pass the raw resources directly — the DevCommResource from
       nccl_comm.create_dev_comm and the RegisteredWindowHandle from
       nccl_comm.register_window. The registered JIT arg adapters
       convert them implicitly, so inside this body the parameters are
       already nccl_cute.DevComm / nccl_cute.Window. Annotate with the
       resource types (nccl.DevCommResource /
       nccl.RegisteredWindowHandle). Empty annotations also run, but
       forfeit the type checking; annotating with the nccl_cute types
       makes CuTeDSL report an error.
    2. Convert explicitly at the call site —
       nccl_cute.DevComm(dev_comm_resource) /
       nccl_cute.Window(win_resource) — and annotate with the nccl_cute
       types. Empty annotations also run, but forfeit the type checking.

    Form 1 saves a line per argument at the call site; form 2 (used
    here) gives better IDE completion and keeps static analysis tools
    happy, since the annotations name the types the body actually sees.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        send_win: CuTeDSL view of the registered source window.
        recv_win: CuTeDSL view of the registered destination window.
    """
    test_nccl_put_kernel(dev_comm, send_win, recv_win).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        cooperative=True
    )


def main():
    """Run a 1 MiB GIN put + signal demo across two MPI ranks.

    Returns:
        Exit code; 0 on success. Host-side validation on the receiver
        prints ``[SUCCESS]`` or ``[ERROR N / NUM_ELEMS mismatches]``.
    """
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    nranks = comm_mpi.Get_size()
    root = 0

    device = Device(rank % system.get_num_devices())
    device.set_current()

    unique_id = nccl.get_unique_id() if rank == root else None
    unique_id = comm_mpi.bcast(unique_id, root=root)

    nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    if rank == root:
        print(f"Running with {nranks} ranks, transferring {NUM_ELEMS * 8} bytes...")

    # Two distinct buffers. Rank 0 fills its send_buf with a pattern; rank 1's
    # recv_buf starts zeroed so we can tell the transfer actually happened.
    # The other (unused) buffer on each rank starts zeroed too — it only
    # exists because window registration is collective.
    send_buf = nccl.cupy.empty(NUM_ELEMS, dtype='int64')
    recv_buf = nccl.cupy.empty(NUM_ELEMS, dtype='int64')
    if rank == 0:
        send_buf[:] = cp.arange(NUM_ELEMS, dtype='int64')
    else:
        send_buf[:] = 0
    recv_buf[:] = 0
    device.sync()  # make host-side fill visible before the kernel runs

    send_win_resource = nccl_comm.register_window(send_buf)
    recv_win_resource = nccl_comm.register_window(recv_buf)
    assert send_win_resource is not None and send_win_resource.is_valid
    assert recv_win_resource is not None and recv_win_resource.is_valid

    reqs = nccl.NCCLDevCommRequirements(
        gin_connection_type=nccl.NcclGinConnectionType.FULL,
        gin_signal_count=SIGNAL_ID + 1,
    )
    dev_comm_resource = nccl_comm.create_dev_comm(requirements=reqs)
    assert dev_comm_resource.is_valid
    assert dev_comm_resource.ptr != 0
    dev_comm = nccl_cute.DevComm(dev_comm_resource)

    send_win = nccl_cute.Window(send_win_resource)
    recv_win = nccl_cute.Window(recv_win_resource)

    # Form 1: raw resources — to use this call, also swap the annotations in
    # test_nccl_put's signature to the commented resource types.
    #test_nccl_put(dev_comm_resource, send_win_resource, recv_win_resource)
    test_nccl_put(dev_comm, send_win, recv_win)

    device.sync()

    # Host-side validation on the receiver — compare the full 1 MiB payload.
    if rank == DST_RANK:
        expected = cp.arange(NUM_ELEMS, dtype='int64')
        mismatches = int((recv_buf != expected).sum().item())
        if mismatches == 0:
            print(f"[rank {rank}] [SUCCESS] {NUM_ELEMS * 8} bytes transferred correctly")
        else:
            print(f"[rank {rank}] [ERROR] {mismatches} / {NUM_ELEMS} mismatches")

    dev_comm_resource.close()
    send_win_resource.close()
    recv_win_resource.close()
    nccl_comm.destroy()

    return 0


if __name__ == "__main__":
    sys.exit(main())
