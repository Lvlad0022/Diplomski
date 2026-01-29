#!/usr/bin/env python3
"""
4 CPU workers → 1 ZMQ collector → SharedExperienceMemory → trainer
Trainer has priority 1, workers 10 → trainer never waits.
"""
import os, time, random, zmq, multiprocessing as mp, psutil
from contextlib import closing
import numpy as np  
from q_logic.q_logic_memory_classes import ExperienceMemory
from environment import SimpleSnakeEnv
from paralell_training_agent import snakeAgent_inference
from pathlib import Path


WORKERS        = 4
COLLECTOR_PORT = 5555
MEMORY_PORT    = 5556

# -------------------------------------------------
# Memory server - runs in its own process
# -------------------------------------------------
def memory_server(capacity=200_000):
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
                batch = None
                if len(memory) > batch_size:
                    batch = memory.sample(batch_size)
                    print(type(batch))
                else:
                    print(f"cannot sample memory is {len(memory)}")
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
            message = sock.recv_pyobj()
            worker_done, memory = message
            if worker_done == True:
                done += 1
                continue
            mem_client.push( memory, 1.0)
        
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
        if batch is not None:
            samples, data_idxs, weights, sample_priorities, sample_log = batch
            print(data_idxs)
            elapsed = (time.time() - t0) * 1000
            print(f"[Trainer] batch-len={len(batch)} sample-time={elapsed:.2f} ms")
        time.sleep(1)
    
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

        #gamelogic
        time.sleep(0.2)
        env = SimpleSnakeEnv(size = 10)
        agent1 = snakeAgent_inference(train= False, noisy_net= True, on_gpu = False)
        
        current_dir = Path(__file__).parent
        file_name = "snakeagent1__polyakTrue_gamma0.99_doubleqTrue_priorityTrue_noisynetTruezero_survive_reward_ver0_2026-01-13_13-12-24.pt.pt"
        model_path = current_dir/ "representative_models" / file_name
        agent1.load_model_state(model_path, noisynet=True, training=False)

        sum_jabuke = 0

        for i in range(100):
            state, snake = env.reset()
            done = False
            count = 0
            jabuka = 0
            reward = 0
            jabuka_novi = 0
            while not done:
                count +=1
                # Random action just to view the game
                if(count <0):
                    action = random.randint(0,3)
                else:
                    action = agent1.get_action((state,snake,reward,jabuka,done))

                if reward >= 0.5:
                    sum_jabuke += 1
                    jabuka_novi += 1

                state_novi,snake_state_novi,reward_novi,done_novi, info = env.step(action)
                memory = agent1.remember((state,snake,reward,jabuka,done),(state_novi,snake_state_novi,reward_novi,jabuka_novi,done_novi))

                state, snake, reward,jabuka,done   =state_novi,snake_state_novi,reward_novi,jabuka_novi,done_novi

                sock.send_pyobj((False, memory))
            print(f"worker{wid} igra {i} jabuka {jabuka_novi}")

        sock.send_pyobj((True, 1))


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


# jos samo trebam shvatit koji je tocno problem s loadanjem modela u agent_inference