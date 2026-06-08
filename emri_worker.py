#!/usr/bin/env python
import os
import sys
import time
import signal
import numpy as np
import MetricMathStreamline as mm
import MainLoopStreamline as ml
import OrbitPlotter as op
from email.utils import parsedate_to_datetime
from multiprocessing import Process, Lock
import argparse
import yaml
import re
import psutil

def last_index_write_time():
    try:
        index = ml.load_index()
    except Exception:
        # index is being written; treat as "just written"
        return time.time()

    if not index:
        return 0.0

    for entry in reversed(index.values()):
        created = entry.get("Created")
        if created is not None:
            try:
                return parsedate_to_datetime(created).timestamp()
            except Exception:
                pass

    return 0.0

def run_single_inspiral(inspiral_cfg, write_poll=20.0, min_spacing=60.0, lock=None):
    """
    inspiral_str : string defining initial EMRI
    write_poll   : seconds to wait when checking if it's safe to write
    min_spacing  : minimum seconds to wait between global writes
    """

    drain_requested = False
    stop_requested = False
    inspiral_str = make_inspiral_str(inspiral_cfg)
    title = inspiral_str.split("label")[1].split("'")[1]
    heartbeat_path = f"D:/EMRIData/inspiral_{title.replace(' ', '_')}.log"

    def heartbeat(msg):
        """Append a message to the worker log immediately."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(heartbeat_path, "a") as f:
            f.write(f"[{timestamp}] - [worker_{pid}] {msg}\n")
        print(f"[{timestamp}] - [worker_{pid}] {msg}", flush=True)

    # To bring everything to a stop and prevent horrible loops!
    def handle_shutdown(signum, frame):
        nonlocal drain_requested, stop_requested
        if not drain_requested:
            heartbeat("Drain signal received, finishing current chunk...")
            drain_requested = True
        else:
            heartbeat("Shutdown signal received, ending current chunk...")
            stop_requested = True


    #def should_stop():
    #    heartbeat(f"Sending shutdown signal...")
    #    return stop_requested
    def should_stop():
        return stop_requested

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    pid = os.getpid()

    heartbeat(f"Started {title}")
    # ---- Load latest checkpoint ONCE ----
    heartbeat("Searching index for pre-existing chunks...")  
    with lock:
        index = ml.load_index()
        
    refs = [
        name for name, dat in index.items()
        if (title in dat["Label"]) and ("copy" not in dat["Label"])
    ]

    if len(refs) == 0:
        heartbeat("No pre-existing chunks found.")
        ins = False

    else:
        heartbeat("Pre-existing chunks found.")
        ins = ml.load_emri_data(refs[-1], quiet=True, reconstruct=False)
        if (ins["plunge"] or ins["unbind"]) or (not ins["stop"] and len(ins["raw"]) < 10**7):
            heartbeat(f"{title} already completed!")
            stop_requested = True

    chunk_counter = len(refs)

    # ---- Main evolution loop ----
    while not stop_requested:
        if drain_requested:
            heartbeat("Drain signal registered, exiting...")
            break
        # Advance trajectory
        heartbeat(f"Starting chunk {chunk_counter + 1}.")
        if not ins:
            ins = eval(inspiral_str)
        else:
            ins = ml.EMRIGenerator(
                ins["inputs"][1],
                ins["inputs"][2],
                endflag=ins["inputs"][3],
                pos=ins["raw"][-1, :4],
                veltrue=ins["raw"][-1, 4:],
                label=f"{title} {chunk_counter + 1}",
                verbose=0,
                err_target=ins["inputs"][4],
                force_stop=should_stop,
            )

        heartbeat(f"Completed chunk {chunk_counter + 1}, attempting to save...")
        time.sleep(np.random.uniform(1, 6))
        chunk_name = ml.save_emri_data(ins, auto=True, lock=lock)   
        heartbeat(f"Chunk {chunk_counter + 1} saved as {chunk_name}.")
        time.sleep(np.random.uniform(1, 5))
        chunk_counter += 1

        if ins.get("stop", False):
            break

        if ((ins["plunge"] or ins["unbind"]) or ins["stop"]) or (not ins["stop"] and len(ins["raw"]) < 10**7):
            heartbeat(f"{title} complete!")
            break

    heartbeat(f"{title} exiting cleanly.")

def format_value(val):
    # Handle lists recursively
    if isinstance(val, list):
        return "[" + ", ".join(format_value(v) for v in val) + "]"

    # Handle strings
    if isinstance(val, str):
        if val == "should_stop" or bool(re.search(r"[+\-*/()**]|np\.", val)):
            return val
        
        # Try to interpret as number
        try:
            num = float(val)
            return repr(num)
        except ValueError:
            return repr(val)

    # Numbers
    if isinstance(val, (int, float)):
        return repr(val)

    # Booleans
    if isinstance(val, bool):
        return "True" if val else "False"

    return repr(val)

def make_inspiral_str(cfg):
    # Go through this list of accepted inputs
    defaults = {"a": 0.0,
                "mu": 0.0,
                "endflag": "min_radius < 0.5", 
                "mass": 1.0, 
                "err_target": 1e-15, 
                "label": "default", 
                "cons": False, 
                "velorient": False, 
                "vel4": False, 
                "params": False, 
                "pos": False, 
                "veltrue": False, 
                "units": "grav",
                "verbose": 0,
                "weird": False,
                "force_stop": "should_stop"
                } 
    
    insp_str = "ml.EMRIGenerator("
    for keyword, value in defaults.items():
        if keyword in cfg.keys():
            value = cfg[keyword]

        f_value = format_value(value)
        if keyword == "label":
            if "'" not in f_value:
                f_value = f"'{f_value}'"
        insp_str += f"{keyword} = {f_value}, "
    insp_str += ")"

    return insp_str

# ----------- Main launcher -----------

def lower_priority(proc):
    try:
        if sys.platform.startswith("win"):
            import psutil
            psutil.Process(proc.pid).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.setpriority(os.PRIO_PROCESS, proc.pid, 10)
    except Exception as e:
        print(f"Priority set failed: {e}", flush=True)

def main():

    main_drain_requested = False
    shutdown_requested = False

    def handle_main_shutdown(signum, frame):
        nonlocal main_drain_requested, shutdown_requested
        if main_drain_requested:
            shutdown_requested = True
            print("\nMain process received shutdown signal.", flush=True)
        else:
            main_drain_requested = True
            print("\nMain process received drain signal; use Ctrl+C again to initiate full shutdown.", flush=True)


    signal.signal(signal.SIGINT, handle_main_shutdown)
    signal.signal(signal.SIGTERM, handle_main_shutdown)

    lock = Lock()

    parser = argparse.ArgumentParser(description="Run EMRI inspirals from YAML config.")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--timeout", type=float, default=30.0)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    inspiral_cfgs = config["inspirals"]

    # stagger write poll intervals to avoid simultaneous writes
    polls = np.random.uniform(0.5, 6, len(inspiral_cfgs))
    polls = np.cumsum(polls)

    max_workers = min(len(inspiral_cfgs), max(1, int(os.cpu_count()//3)))
    print(f"Preparing {max_workers} workers.", flush=True)

    processes = []
    active = []

    for cfg, p in zip(inspiral_cfgs, polls):
        while len(active) >= max_workers and not shutdown_requested:
            for proc in active[:]:
                if not proc.is_alive():
                    proc.join()
                    active.remove(proc)
            time.sleep(0.5)

        if shutdown_requested or main_drain_requested:
            break
        
        proc = Process(target=run_single_inspiral, args=(cfg, p, args.timeout, lock))
        proc.start()
        lower_priority(proc)
        processes.append(proc)
        active.append(proc)
    
    for proc in active[:]:
        while proc.is_alive():
            if shutdown_requested:
                break
            time.sleep(0.2)
        proc.join()

    if shutdown_requested:
        print("Shutting down workers...", flush=True)

        for proc in active[:]:
            if proc.is_alive():
                proc.terminate()

        for proc in active:
            proc.join()

        print("All workers terminated.")

if __name__ == "__main__":
    main()
