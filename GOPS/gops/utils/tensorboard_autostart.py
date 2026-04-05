import os
import socket
import subprocess
import sys


def _is_port_in_use(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        return sock.connect_ex((host, int(port))) == 0
    finally:
        sock.close()


def _find_free_port(host, start_port, max_tries=30):
    start_port = int(start_port)
    for offset in range(max_tries):
        cand = start_port + offset
        if not _is_port_in_use(host, cand):
            return cand
    return None


def maybe_start_tensorboard(args):
    if not args.get("auto_tensorboard", False):
        return

    host = args.get("tensorboard_host", "0.0.0.0")
    port = int(args.get("tensorboard_port", 6006))
    logdir_spec = f"run1:{args['save_folder']}"

    if _is_port_in_use("127.0.0.1", port):
        new_port = _find_free_port("127.0.0.1", port + 1)
        if new_port is None:
            print(
                f"[TensorBoard] port {port} is already in use, and no free "
                f"fallback port was found. Existing UI may point to an old run."
            )
            return
        print(
            f"[TensorBoard] port {port} is already in use. "
            f"Starting current run on port {new_port}."
        )
        port = new_port

    tb_cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir_spec",
        logdir_spec,
        "--host",
        host,
        "--port",
        str(port),
    ]
    tb_env = os.environ.copy()
    # Avoid loading incompatible user-site packages from ~/.local.
    tb_env["PYTHONNOUSERSITE"] = "1"

    subprocess.Popen(
        tb_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=tb_env,
        start_new_session=True,
    )
    print(f"[TensorBoard] started at http://localhost:{port}, run=run1")
