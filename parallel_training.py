#!/usr/bin/env python3
"""
4 CPU workers → 1 ZMQ collector → SharedExperienceMemory → trainer
Trainer has priority 1, workers 10 → trainer never waits.
"""
import os, time, random, zmq, multiprocessing as mp, psutil
from contextlib import closing
import numpy as np  
from q_logic.q_logic_memory_classes import ExperienceMemory

WORKERS        = 4
COLLECTOR_PORT = 5555
MEMORY_PORT    = 5556

# -------------------------------------------------
# Memory server - runs in its own process
# -------------------------------------------------
def memory_server(capacity=50_000):
    """Dedicated process hosting the ExperienceMemory."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{MEMORY_PORT}")
    
    memory = ExperienceMemory(capacity=capacity, priorities=True)
    print(f"[Memory] server ready on port {MEMORY_PORT}")
    
    while True:
        try:
            cmd, *args = sock.recv_pyobj()
            
            if cmd == "push":
                exp, prio = args
                memory.push(exp, prio)
                sock.send_pyobj("OK")
                
            elif cmd == "sample":
                batch_size, = args
                batch = memory.sample(batch_size)
                sock.send_pyobj(batch)
                
            elif cmd == "update":
                idxs, td, priorities = args
                memory.update_priorities(idxs, td, priorities)
                sock.send_pyobj("OK")
                
            elif cmd == "len":
                sock.send_pyobj(len(memory))
                
            elif cmd == "shutdown":
                sock.send_pyobj(len(memory))
                break
                
        except Exception as e:
            print(f"[Memory] error: {e}")
            sock.send_pyobj(None)
    
    sock.close()
    ctx.term()
    print("[Memory] server shutdown")


# -------------------------------------------------
# Memory client - used by other processes
# -------------------------------------------------
class MemoryClient:
    """Client interface to the memory server."""
    def __init__(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(f"tcp://127.0.0.1:{MEMORY_PORT}")
        time.sleep(0.1)  # let connection establish
    
    def push(self, exp, priority=1.0):
        self.sock.send_pyobj(("push", exp, priority))
        return self.sock.recv_pyobj()
    
    def sample(self, batch_size):
        self.sock.send_pyobj(("sample", batch_size))
        return self.sock.recv_pyobj()
    
    def update_priorities(self, idxs, td, priorities):
        self.sock.send_pyobj(("update", idxs, td, priorities))
        return self.sock.recv_pyobj()
    
    def __len__(self):
        self.sock.send_pyobj(("len",))
        return self.sock.recv_pyobj()
    
    def shutdown(self):
        self.sock.send_pyobj(("shutdown",))
        size = self.sock.recv_pyobj()
        self.sock.close()
        self.ctx.term()
        return size
    
    def close(self):
        self.sock.close()
        self.ctx.term()


# -------------------------------------------------
# Collector
# -------------------------------------------------
def collector():
    ctx = zmq.Context()
    mem_client = MemoryClient()
    
    with closing(ctx.socket(zmq.PULL)) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        while True:
            try:
                sock.bind(f"tcp://127.0.0.1:{COLLECTOR_PORT}")
                break
            except zmq.ZMQError:
                time.sleep(0.5)
        
        print("[Collector] ready – waiting for workers …")
        done = 0
        while done < WORKERS:
            wid, game_no, pos = sock.recv_json()
            if game_no == "DONE":
                done += 1
                continue
            mem_client.push((wid, game_no, pos), 1.0)
        
        print("[Collector] all workers finished")
    
    mem_client.close()


# -------------------------------------------------
# Trainer
# -------------------------------------------------
def trainer():
    print("[Trainer] started")
    mem_client = MemoryClient()
    time.sleep(1.0)
    
    for _ in range(10):
        t0 = time.time()
        batch = mem_client.sample(5)
        elapsed = (time.time() - t0) * 1000
        print(f"[Trainer] batch-len={len(batch)} sample-time={elapsed:.2f} ms")
        time.sleep(0.1)
    
    print("[Trainer] finished")
    mem_client.close()


# -------------------------------------------------
# Worker
# -------------------------------------------------
def worker(wid):
    try:
        psutil.Process(os.getpid()).cpu_affinity([wid + 1])
    except AttributeError:
        pass

    ctx = zmq.Context()
    with closing(ctx.socket(zmq.PUSH)) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://127.0.0.1:{COLLECTOR_PORT}")
        time.sleep(0.2)
        
        t0 = time.time()
        sent = 0
        while time.time() - t0 < 5:
            for _ in range(50):
                pos = random.randint(-100, 100)
                sock.send_json((wid, sent, pos))
                sent += 1
            time.sleep(1)
        sock.send_json((wid, "DONE", None))


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Start memory server first
    mem_proc = mp.Process(target=memory_server, args=(50_000,), daemon=False)
    mem_proc.start()
    time.sleep(0.5)  # let server start

    coll_proc = mp.Process(target=collector, daemon=False)
    train_proc = mp.Process(target=trainer, daemon=False)

    coll_proc.start()
    train_proc.start()

    workers = [mp.Process(target=worker, args=(i,)) for i in range(WORKERS)]
    for w in workers:
        w.start()

    try:
        for p in workers + [coll_proc, train_proc]:
            p.join()
    except KeyboardInterrupt:
        print("\nCtrl-C – terminating")
        for p in workers + [coll_proc, train_proc]:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

    # Shutdown memory server and get final size
    client = MemoryClient()
    final_size = client.shutdown()
    mem_proc.join(timeout=2)
    
    print(f"\nAll done. Memory size: {final_size}")