#!/usr/bin/env python
import os
import time
import signal
import numpy as np
import MetricMathStreamline as mm
import MainLoopStreamline as ml
import OrbitPlotter as op
from email.utils import parsedate_to_datetime
from multiprocessing import Process

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


def run_single_inspiral(inspiral_str, write_poll=20.0, min_spacing=60.0):
    """
    inspiral_str : string defining initial EMRI
    write_poll   : seconds to wait when checking if it's safe to write
    min_spacing  : minimum seconds to wait between global writes
    """

    stop_requested = False
    title = inspiral_str.split('"')[-2]
    heartbeat_path = f"D:/EMRIData/inspiral_{title.replace(' ', '_')}.log"

    def heartbeat(msg):
        """Append a message to the worker log immediately."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(heartbeat_path, "a") as f:
            f.write(f"[{timestamp}] - [worker_{pid}] {msg}\n")
            f.flush()
        print(f"[{timestamp}] - [worker_{pid}] {msg}", flush=True)

    # To bring everything to a stop and prevent horrible loops!
    def handle_shutdown(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        heartbeat("Shutdown signal received, ending current chunk...")

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
    index = ml.load_index()
    refs = [
        name for name, dat in index.items()
        if (title in dat["Label"]) and ("copy" not in dat["Label"])
    ]

    if len(refs) == 0:
        heartbeat("No pre-existing chunks found, starting chunk 1.")
        ins = eval(inspiral_str)
        heartbeat(f"Completed chunk 1, saving...")
        refs.append(ml.save_emri_data(ins, auto=True))
    else:
        heartbeat("Pre-existing chunks found.")
        ins = ml.load_emri_data(refs[-1], quiet=True)

    chunk_counter = max(1, len(refs))

    # ---- Main evolution loop ----
    while not stop_requested:
        if ins["plunge"]:
            heartbeat(f"{title} complete!")
            break

        # Advance trajectory
        heartbeat(f"Starting chunk {chunk_counter + 1}.")
        ins = ml.EMRIGenerator(
            ins["spin"],
            1e-4,
            pos=ins["raw"][-1, :4],
            veltrue=ins["raw"][-1, 4:],
            label=f"{title} {chunk_counter + 1}",
            verbose=0,
            err_target=ins["inputs"][4],
            force_stop=should_stop,
        )

        heartbeat(f"Completed chunk {chunk_counter + 1}, attempting to save...")
        time.sleep(write_poll)
        while True:
            last_write = last_index_write_time()
            now = time.time()
            if now - last_write >= min_spacing:
                break
            heartbeat(f"Last write to index too recent, waiting...")
            time.sleep(write_poll)

        chunk_name = ml.save_emri_data(ins, auto=True)
        heartbeat(f"Chunk {chunk_counter + 1} saved as {chunk_name}.")
        chunk_counter += 1

        if ins.get("stop", False):
            break

    heartbeat(f"{title} exiting cleanly.")


# ----------- Main launcher -----------

def main():
    inspiral_strs = [
        'ml.EMRIGenerator(-0.9, 1e-4, params=[10, 1e-3, 60*np.pi/180], label="Quasi-Circular Inspiral (Paper)", err_target=1e-12, verbose=0, force_stop=should_stop)',
        'ml.EMRIGenerator(0.0, 1e-5, cons=[0.99410372,3.9525, 0], label="Eq. Zoom-Whirl Inspiral (Paper)", err_target=1e-15, verbose=0, force_stop=should_stop)',
        'ml.EMRIGenerator(-0.8, 1e-4, params=[30/(1 - 0.4**2), 0.4, 30*np.pi/180], label="Generic Inspiral (Paper)", err_target=1e-12, verbose=0, force_stop=should_stop)',
        'ml.EMRIGenerator(0.95, 1e-4, cons=[0.962076494,0.31252652, 12.4], label="Near-Polar Inspiral (Paper)", err_target=1e-12, verbose=0, force_stop=should_stop)'
    ]

    # stagger write poll intervals to avoid simultaneous writes
    polls = np.random.uniform(0.5, 6, len(inspiral_strs))
    polls = np.cumsum(polls)

    print("Preparing workers.", flush=True)

    max_workers = min(len(inspiral_strs), os.cpu_count() - 1)

    processes = []
    for s, p in zip(inspiral_strs, polls):
        proc = Process(target=run_single_inspiral, args=(s, p, 30.0))
        processes.append(proc)
        proc.start()

    try:
        for proc in processes:
            proc.join()
    except KeyboardInterrupt:
        for proc in processes:
            proc.join()

if __name__ == "__main__":
    main()
