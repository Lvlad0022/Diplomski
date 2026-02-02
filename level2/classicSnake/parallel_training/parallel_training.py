import os, time, random, zmq, multiprocessing as mp, psutil
from contextlib import closing
from collections import deque
import threading
import numpy as np  
from q_logic.q_logic_memory_classes import TDPriorityReplayBuffer
from environment import SimpleSnakeEnv
from Diplomski.level2.classicSnake.parallel_training.paralell_training_agent import snakeAgent_inference, snakeAgent_trainer
from pathlib import Path
from q_logic.q_logic import set_seed


WORKERS = 4
COLLECTOR_PORT = 5555
MEMORY_PORT = 5556
MODEL_SYNC_PORT = 5557

# -------------------------------------------------
# Memory server - UNCHANGED
# -------------------------------------------------
def memory_server(capacity=200_000):
    """Dedicated process hosting the ExperienceMemory."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{MEMORY_PORT}")
    
    memory = TDPriorityReplayBuffer(capacity=capacity)
    print(f"[Memory] server ready on port {MEMORY_PORT}")
    
    while True:
        try:
            cmd, *args = sock.recv_pyobj()
            
            if cmd == "push":
                exp, prio = args
                num_visits, td_errors = memory.push(exp)
                sock.send_pyobj("OK")
                
            elif cmd == "sample":
                batch_size, = args
                batch = None
                if len(memory) > batch_size:
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
# Memory client - Basic version
# -------------------------------------------------
class MemoryClient:
    """Client interface to the memory server."""
    def __init__(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(f"tcp://127.0.0.1:{MEMORY_PORT}")
        time.sleep(0.1)
    
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
# Async batch prefetcher - FIXED with separate socket
# -------------------------------------------------
class AsyncBatchPrefetcher:
    """Prefetches batches in background thread with its own socket."""
    def __init__(self, batch_size, prefetch_count=2):
        self.batch_size = batch_size
        self.prefetch_count = prefetch_count
        
        self.queue = deque()
        self.lock = threading.Lock()
        self.active = True
        
        # Create separate socket for prefetching thread
        self.ctx = zmq.Context()
        self.sock = None
        
        self.thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()
    
    def _prefetch_loop(self):
        """Background thread with its own ZMQ socket."""
        # Initialize socket in the thread
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(f"tcp://127.0.0.1:{MEMORY_PORT}")
        time.sleep(0.1)
        
        while self.active:
            with self.lock:
                queue_size = len(self.queue)
            
            if queue_size < self.prefetch_count:
                try:
                    # Use thread's own socket
                    self.sock.send_pyobj(("sample", self.batch_size))
                    batch = self.sock.recv_pyobj()
                    
                    if batch is not None:
                        with self.lock:
                            self.queue.append(batch)
                except Exception as e:
                    print(f"[Prefetcher] error: {e}")
                    time.sleep(0.01)
            else:
                time.sleep(0.001)
        
        # Cleanup
        self.sock.close()
    
    def get_batch(self):
        """Get next batch (blocking if queue empty)."""
        while self.active:
            with self.lock:
                if self.queue:
                    return self.queue.popleft()
            time.sleep(0.001)
        return None
    
    def stop(self):
        """Stop prefetching."""
        self.active = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        self.ctx.term()


# -------------------------------------------------
# Model broadcaster - UNCHANGED
# -------------------------------------------------
class ModelBroadcaster:
    """Trainer uses this to broadcast model updates."""
    def __init__(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://127.0.0.1:{MODEL_SYNC_PORT}")
        time.sleep(0.5)
    
    def broadcast_model(self, model_state_dict):
        self.sock.send_pyobj(("model_update", model_state_dict))
    
    def broadcast_shutdown(self):
        self.sock.send_pyobj(("shutdown", None))
    
    def close(self):
        self.sock.close()
        self.ctx.term()


# -------------------------------------------------
# Model subscriber - UNCHANGED
# -------------------------------------------------
class ModelSubscriber:
    """Workers use this to receive model updates."""
    def __init__(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(f"tcp://127.0.0.1:{MODEL_SYNC_PORT}")
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.RCVTIMEO, 1)
    
    def check_for_update(self):
        try:
            return self.sock.recv_pyobj()
        except zmq.Again:
            return None
    
    def close(self):
        self.sock.close()
        self.ctx.term()


# -------------------------------------------------
# Collector - UNCHANGED
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
            worker_done, data = sock.recv_pyobj()
            
            if worker_done:
                done += 1
                print(f"[Collector] worker finished ({done}/{WORKERS})")
                continue
            
            mem_client.push(data, 1.0)
        
        print("[Collector] all workers finished")
    
    mem_client.close()


# -------------------------------------------------
# Trainer with async prefetching - FIXED
# -------------------------------------------------
def trainer(sync_every_k=10):
    print("[Trainer] started")
    
    # Separate client for priority updates
    mem_client = MemoryClient()
    broadcaster = ModelBroadcaster()
    
    # Initialize prefetcher with its own connection
    prefetcher = AsyncBatchPrefetcher(batch_size=64, prefetch_count=3)
    time.sleep(0.5)
    
    set_seed(42)
    agent_trainer = snakeAgent_trainer()
    
    train_cycle = 0
    vrijeme_pocetak = time.time()
    fetch_time = 0.0
    train_time = 0.
    update_time = 0.0
    sync_time = 0.0
    br = 0
    ukupno_vrijeme = time.time()
    for _ in range(100_000):
        v = time.time()
        batch = prefetcher.get_batch()
        fetch_time += time.time() - v
        
        if batch is None:
            break
            
        br += 1
        samples, data_idxs, weights, sample_priorities, sample_log = batch
        
        # Training
        t = time.time()
        loss, idxs, td_error, sample_priorities = agent_trainer.train(
            samples, data_idxs, weights, sample_priorities, sample_log
        )
        train_time += time.time() - t
        
        # Priority update (using separate socket)
        u = time.time()
        mem_client.update_priorities(data_idxs, td_error, sample_priorities)
        update_time += time.time() - u
        
        if br % 50 == 0:
            total_time = (time.time() - vrijeme_pocetak) * 1000
            print(f"[Trainer] cycle={train_cycle} batch={len(samples)} "
                  f"total={total_time:.2f}ms fetch={fetch_time*1000:.2f}ms "
                  f"train={train_time*1000:.2f}ms update={update_time*1000:.2f}ms "
                  f"sync={sync_time*1000:.2f}ms "
                  f"lr={agent_trainer.get_current_lr():.6f} loss={loss:.4f}")
            vrijeme_pocetak = time.time()
            fetch_time = 0.0
            train_time = 0.0
            update_time = 0.0
            sync_time = 0.0
        train_cycle += 1
        

        if train_cycle % sync_every_k == 0:
            u = time.time()
            model_state = agent_trainer.get_model_state_dict()
            broadcaster.broadcast_model(model_state)
            sync_time += time.time() - u
    
    prefetcher.stop()
    broadcaster.broadcast_shutdown()
    print(f"[Trainer] total training time: {(time.time() - ukupno_vrijeme):.2f}s")
    print("[Trainer] finished")
    broadcaster.close()
    mem_client.close()


# -------------------------------------------------
# Worker - UNCHANGED
# -------------------------------------------------
def worker(wid):
    try:
        psutil.Process(os.getpid()).cpu_affinity([wid + 1])
    except AttributeError:
        pass

    ctx = zmq.Context()
    model_sub = ModelSubscriber()
    
    with closing(ctx.socket(zmq.PUSH)) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://127.0.0.1:{COLLECTOR_PORT}")
        time.sleep(0.2)

        set_seed(100 + wid)
        env = SimpleSnakeEnv(size=10)
        NoisyNet = True
        isTraining = True
        agent1 = snakeAgent_inference(train=isTraining, noisy_net=NoisyNet, on_gpu=False)

        sum_jabuka = 0
        sum_koraka = 0
        vrijeme_pocetak = time.time()
        shutdown_received = False
        game_count = 0
        
        while not shutdown_received:
            game_count += 1
            state, snake = env.reset()
            done = False
            count = 0
            jabuka = 0
            reward = 0
            jabuka_novi = 0
            
            while not done:
                count += 1
                
                
                if isTraining and NoisyNet:
                    action, ratios = agent1.get_action((state, snake, reward, jabuka, done))
                else:
                    action = agent1.get_action((state,state,reward,jabuka,done))
                

                state_novi, snake_state_novi, reward_novi, done_novi, info = env.step(action)
                memory = agent1.remember(
                    (state, snake, reward, jabuka, done),
                    (state_novi, snake_state_novi, reward_novi, jabuka_novi, done_novi)
                )

                if reward >= 0.5:
                    jabuka_novi += 1
                if count > 500:
                    done_novi = True

                state, snake, reward, jabuka, done = state_novi, snake_state_novi, reward_novi, jabuka_novi, done_novi
                

                if isinstance(memory, list):
                    for m in memory:
                        sock.send_pyobj((False, m))
                else:
                    sock.send_pyobj((False, memory))

            update = model_sub.check_for_update()
            if update:
                cmd, data = update
                if cmd == "model_update":
                    agent1.load_model_state_dict(data, noisynet=NoisyNet, training=isTraining)
                elif cmd == "shutdown":
                    shutdown_received = True
                    break
            
            sum_jabuka += jabuka_novi
            sum_koraka += count
            if game_count % 500 == 0:
                print(f"worker: {wid} igra {game_count} jabuka {sum_jabuka/500} "
                      f"koraka {sum_koraka/500} vrijeme {time.time()-vrijeme_pocetak:.2f}s")
                vrijeme_pocetak = time.time()
                sum_jabuka = 0
                sum_koraka = 0
                
        sock.send_pyobj((True, 1))
        model_sub.close()


# -------------------------------------------------
# Main - UNCHANGED
# -------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    mem_proc = mp.Process(target=memory_server, args=(200_000,), daemon=False)
    mem_proc.start()
    time.sleep(0.5)

    coll_proc = mp.Process(target=collector, daemon=False)
    train_proc = mp.Process(target=trainer, kwargs={"sync_every_k": 10}, daemon=False)

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

    client = MemoryClient()
    final_size = client.shutdown()
    mem_proc.join(timeout=2)
    
    print(f"\nAll done. Memory size: {final_size}")