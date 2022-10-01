from __future__ import print_function
import logging
import multiprocessing
import os
import signal
import subprocess
import sys

CPU_COUNT = multiprocessing.cpu_count()

MODEL_SERVER_TIMEOUT = os.environ.get("MODEL_SERVER_TIMEOUT", 300)
MODEL_SERVER_WORKERS = int(os.environ.get("MODEL_SERVER_WORKERS", CPU_COUNT))

log = logging.getLogger(__name__)


def sigterm_handler(tfs_pid, nginx_pid, gunicorn_pid):
    try:
        os.kill(tfs_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(nginx_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(gunicorn_pid, signal.SIGQUIT)
    except OSError:
        pass

    sys.exit(0)


def start_server():
    log.info(f"Starting the inference server with {MODEL_SERVER_WORKERS} workers...")

    subprocess.check_call(["ln", "-sf", "/dev/stdout", "/var/log/nginx/access.log"])
    subprocess.check_call(["ln", "-sf", "/dev/stderr", "/var/log/nginx/error.log"])

    tfs = subprocess.Popen(["/usr/bin/tf_serving_entrypoint.sh"])
    nginx = subprocess.Popen(["nginx", "-c", "/opt/program/nginx.conf"])
    gunicorn = subprocess.Popen(
        [
            "gunicorn",
            "--timeout",
            str(MODEL_SERVER_TIMEOUT),
            "-k",
            "gevent",
            "-b",
            "unix:/tmp/gunicorn.sock",
            "-w",
            str(MODEL_SERVER_WORKERS),
            "wsgi:app",
        ]
    )

    signal.signal(
        signal.SIGTERM, lambda a, b: sigterm_handler(tfs.pid, nginx.pid, gunicorn.pid)
    )

    pids = {tfs.pid, nginx.pid, gunicorn.pid}
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break

    sigterm_handler(tfs.pid, nginx.pid, gunicorn.pid)
    print("Inference server exiting...")


if __name__ == "__main__":
    start_server()
