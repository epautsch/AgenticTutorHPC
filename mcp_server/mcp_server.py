import os
import socket
import subprocess
from pydantic import BaseModel
import paramiko
import re
from fastapi import FastAPI
import uvicorn
import getpass
username = getpass.getuser()


class RunRequest(BaseModel):
    command: str
    workdir: str



GPU_NODES = ["gpu1", "gpu2"]

ALL_NODES = [
    "gpu1", "gpu2",
    "node2", "node3", "node4", "node5", "node6",
    "node7", "node8", "node9",
    "node11", "node12", "node13", "node14", "node15"
]

# core
def ssh_run_cmd(hostname, command, username=None):
    """
    Runs a command on a remote node using SSH and returns stdout.
    Username is auto-detected unless explicitly provided.
    """
    if username is None:
        username = getpass.getuser()

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname, username=username)
        stdin, stdout, stderr = client.exec_command(command)
        output = stdout.read().decode()
        error = stderr.read().decode()
        client.close()

        if error.strip():
            return None, error
        return output, None
    except Exception as e:
        return None, str(e)

# get gpu util
def get_gpu_utilization(hostname):
    """
    SSH into a GPU node and collect GPU utilization and memory usage.
    Returns a list of tuples: [(util, mem), ...] one per GPU.
    """
    cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"
    out, err = ssh_run_cmd(hostname, cmd)

    if err or out is None:
        return None, err

    gpu_stats = []
    for line in out.strip().split("\n"):
        util_str, mem_str = line.split(",")
        gpu_stats.append((int(util_str.strip()), int(mem_str.strip())))

    return gpu_stats, None

# compute score
def compute_node_load(gpu_stats):
    """
    Given a list of (util, mem) tuples for each GPU, compute a single score.
    Lower score = less loaded = better.
    """
    if not gpu_stats:
        return float("inf")  # treat as super-busy if no data

    # Simple load score: average utilization + small weight for memory usage
    avg_util = sum(util for util, _ in gpu_stats) / len(gpu_stats)
    avg_mem  = sum(mem for _, mem in gpu_stats) / len(gpu_stats)

    # Combine into load score
    score = avg_util + (avg_mem / 1000)  # 1000 MB = 1 utilization point
    return score


app = FastAPI()

# end points
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/list_nodes")
def list_nodes():
    return {"nodes": ALL_NODES}

@app.get("/best_node")
def best_node():
    """
    Determine which GPU node has the lowest GPU load.
    """
    best = None
    best_score = float("inf")
    details = {}

    for node in GPU_NODES:
        gpu_stats, err = get_gpu_utilization(node)
        if err:
            details[node] = {"error": err}
            continue

        score = compute_node_load(gpu_stats)
        details[node] = {
            "gpu_stats": gpu_stats,
            "score": score
        }

        if score < best_score:
            best_score = score
            best = node

    return {
        "best_node": best,
        "details": details
    }


@app.post("/run_on_best_node")
def run_on_best_node(req: RunRequest):
    # Determine best node
    best_info = best_node()
    node = best_info["best_node"]

    if node is None:
        return {"error": "No available GPU nodes"}

    # Build SSH command
    ssh_cmd = f"cd {req.workdir} && {req.command}"

    stdout, stderr = ssh_run_cmd(node, ssh_cmd)

    return {
        "node": node,
        "stdout": stdout,
        "stderr": stderr
    }


def free_port(port):
    """
    Kills any process currently using the specified port.
    """
    try:
        # Find process using the port
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                print(f"Killing process {pid} using port {port}")
                os.kill(int(pid), 9)
    except Exception as e:
        print("Error freeing port:", e)



if __name__ == "__main__":
    free_port(8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
