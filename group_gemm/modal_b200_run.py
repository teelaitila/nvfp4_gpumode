# modal_b200_run.py
import modal
from pathlib import Path

app = modal.App("nvfp4-group-gemm-b200", include_source=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/NVIDIA/cutlass.git",
        "pip install -r cutlass/python/CuTeDSL/requirements.txt",
    )
    .pip_install("torch==2.9.1")
)

# Use a network file system as a lightweight "mount".
# We upload the local folder before invoking the function.
nfs = modal.NetworkFileSystem.from_name(
    "nvfp4-group-gemm-nfs", create_if_missing=True
)

@app.function(
    gpu="B200",
    image=image,
    network_file_systems={"/root/workspace": nfs},
    timeout=60 * 10,
)
def run_kernel():
    import os
    import sys
    import torch

    # Use the uploaded workspace
    os.chdir("/root/workspace/group_gemm")
    sys.path.insert(0, "/root/workspace/group_gemm")
    if not os.path.exists("/root/workspace/group_gemm/reference.py"):
        print("workspace contents:", os.listdir("/root/workspace"))
        print("group_gemm contents:", os.listdir("/root/workspace/group_gemm"))

    from reference import generate_input
    from submission import custom_kernel

    # Minimal single-group test (smallest representative)
    data = generate_input(
        m=[80],
        n=[4096],
        k=[7168],
        g=1,
        seed=1111,
    )

    # Debug breadcrumbs around the kernel call
    print("about to run custom_kernel")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = custom_kernel(data)
    end.record()
    print("custom_kernel returned, waiting for completion")
    # Poll for completion to avoid hard deadlock on synchronize.
    import time
    timeout_s = 15.0
    start_poll = time.time()
    while True:
        if end.query():
            break
        if time.time() - start_poll > timeout_s:
            print("kernel did not complete within timeout")
            return
        time.sleep(0.1)
    torch.cuda.synchronize()
    print("sync complete")

    # Sanity prints
    print("Output groups:", len(out))
    print("Output shape:", out[0].shape)
    print("dtype:", out[0].dtype)
    print("kernel ms:", start.elapsed_time(end))
    print("done")

@app.local_entrypoint()
def main():
    # Upload local workspace to the NFS before running remotely.
    local_dir = Path(__file__).resolve().parent
    nfs.add_local_dir(
        str(local_dir),
        remote_path="/group_gemm",
    )
    run_kernel.remote()