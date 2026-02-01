import os, time, random, zmq, multiprocessing as mp, psutil
from contextlib import closing
import numpy as np  
from q_logic.q_logic_memory_classes import TDPriorityReplayBuffer
from environment import SimpleSnakeEnv
from paralell_training_agent import snakeAgent_inference, snakeAgent_trainer
from pathlib import Path
from q_logic.q_logic import set_seed


WORKERS        = 4
COLLECTOR_PORT = 5555
MEMORY_PORT    = 5556
MODEL_SYNC_PORT = 5557  # New port for model broadcasting

# -------------------------------------------------
# Memory server - runs in its own process
# -------------------------------------------------
def memory_server(capacity=200_000):
    """Dedicated process hosting the ExperienceMemory."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.R)
    sock.bind(f"tcp://127.0.0.1:{MEMORY_PORT}")
    
    memory = TDPriorityReplayBuffer(capacity=capacity)
    print(f"[Memory] server ready on port {MEMORY_PORT}")
    
    while True:
        try:
            cmd, *args = sock.recv_pyobj()
            
            if cmd == "push":
                exp, prio = args
                num_visits, td_errors = memory.push(exp)
  #              if num_visits is not None:
   #                 print(f"(visits: {num_visits}, td_error: {td_errors:.4f})")
                sock.send_pyobj("OK")
                
            elif cmd == "sample":
                batch_size, = args
                batch = None
                if len(memory) > batch_size:
                    batch = memory.sample(batch_size)
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
# Model broadcaster for trainer
# -------------------------------------------------
class ModelBroadcaster:
    """Trainer uses this to broadcast model updates."""
    def __init__(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://127.0.0.1:{MODEL_SYNC_PORT}")
        time.sleep(0.5)  # Let subscribers connect
    
    def broadcast_model(self, model_state_dict):
        """Send model state dict to all workers."""
        self.sock.send_pyobj(("model_update", model_state_dict))
        #print("[Broadcaster] Model update sent")
    
    def broadcast_shutdown(self):
        """Signal workers to stop."""
        self.sock.send_pyobj(("shutdown", None))
    
    def close(self):
        self.sock.close()
        self.ctx.term()


# -------------------------------------------------
# Model subscriber for workers
# -------------------------------------------------
class ModelSubscriber:
    """Workers use this to receive model updates."""
    def __init__(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(f"tcp://127.0.0.1:{MODEL_SYNC_PORT}")
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        self.sock.setsockopt(zmq.RCVTIMEO, 1)  # 100ms timeout
    
    def check_for_update(self):
        """Non-blocking check for model update. Returns (cmd, data) or None."""
        try:
            return self.sock.recv_pyobj()
        except zmq.Again:
            return None
    
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
            worker_done, data = sock.recv_pyobj()
            
            if worker_done:
                done += 1
                print(f"[Collector] worker finished ({done}/{WORKERS})")
                continue
            
            mem_client.push(data, 1.0)
        
        print("[Collector] all workers finished")
    
    mem_client.close()


# -------------------------------------------------
# Trainer
# -------------------------------------------------
def trainer(sync_every_k=10):
    print("[Trainer] started")
    mem_client = MemoryClient()
    broadcaster = ModelBroadcaster()
    time.sleep(0.01)
    
    set_seed(42)
    agent_trainer = snakeAgent_trainer()

    
    train_cycle = 0
    vrijeme_pocetak = time.time()
    fetch_time = 0.0
    br = 0
    for _ in range(10000):  # Example: 100 training iterations

        v = time.time()
        batch = mem_client.sample(64)  # Larger batch for training
        fetch_time += time.time() - v
        
        if batch is not None:
            br+=1
            samples, data_idxs, weights, sample_priorities, sample_log = batch
            loss , idxs, td_error, sample_priorities =agent_trainer.train(samples, data_idxs, weights, sample_priorities, sample_log)
            mem_client.update_priorities(data_idxs, td_error, sample_priorities)
            

            if br%50 == 0:
                print(f"[Trainer] cycle={train_cycle} batch-len={len(samples)} time={(time.time() - vrijeme_pocetak)*1000:.2f}ms fetch_time={(fetch_time)*1000:.2f}ms lr={agent_trainer.get_current_lr():.6f} loss={loss:.4f}")
                vrijeme_pocetak = time.time()
                fetch_time = 0.0
            
            train_cycle += 1
            
            # Sync model to workers every k cycles
            if train_cycle % sync_every_k == 0:
                # Get model state dict (depends on your agent implementation)
                # model_state = agent.get_model_state_dict()
                model_state =  agent_trainer.get_model_state_dict()
                broadcaster.broadcast_model(model_state)
                #print(f"[Trainer] Synced model at cycle {train_cycle}")
        
    
    # Signal shutdown
    broadcaster.broadcast_shutdown()
    print("[Trainer] finished")
    broadcaster.close()
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
    model_sub = ModelSubscriber()
    
    with closing(ctx.socket(zmq.PUSH)) as sock:
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://127.0.0.1:{COLLECTOR_PORT}")
        time.sleep(0.2)

        set_seed(100 + wid)
        # Game logic
        env = SimpleSnakeEnv(size=10)
        agent1 = snakeAgent_inference(train=False, noisy_net=True, on_gpu=False)
        
        current_dir = Path(__file__).parent
        #file_name = "snakeagent1__polyakTrue_gamma0.99_doubleqTrue_priorityTrue_noisynetTruezero_survive_reward_ver0_2026-01-13_13-12-24.pt.pt"
        #model_path = current_dir / "representative_models" / file_name
        #agent1.load_model_state(model_path, noisynet=True, training=False)

        sum_jabuka = 0
        sum_koraka = 0
        vrijeme_pocetak = time.time()
        shutdown_received = False
        game_count = 0
        while not shutdown_received:
            if shutdown_received:
                break
            
            game_count += 1
            state, snake = env.reset()
            done = False
            count = 0
            jabuka = 0
            reward = 0
            jabuka_novi = 0

            
            while not done:
                # Check for model updates (non-blocking)
                
                count += 1
                
                if count < 0:
                    action = random.randint(0, 3)
                else:
                    action = agent1.get_action((state, snake, reward, jabuka, done))

                if reward >= 0.5:
                    jabuka_novi += 1

                state_novi, snake_state_novi, reward_novi, done_novi, info = env.step(action)
                memory = agent1.remember(
                    (state, snake, reward, jabuka, done),
                    (state_novi, snake_state_novi, reward_novi, jabuka_novi, done_novi)
                )

                state, snake, reward, jabuka, done = state_novi, snake_state_novi, reward_novi, jabuka_novi, done_novi

                if isinstance(memory, list) :
                    for m in memory:
                        sock.send_pyobj((False, m))
                else:
                    sock.send_pyobj((False, memory))

            a = time.time()
            update = model_sub.check_for_update()
            if update:
                cmd, data = update
                if cmd == "model_update":
                    agent1.load_model_state_dict(data, noisynet=True, training=False)
                    #print(f"[Worker {wid}] update time: {time.time() - a:.4f}s")
                elif cmd == "shutdown":
                    shutdown_received = True
                    break
            
            sum_jabuka += jabuka_novi
            sum_koraka += count
            if game_count % 500 == 0:
                print(f"worker: {wid} igra {game_count} jabuka {sum_jabuka/500} koraka {sum_koraka/500} vrijeme {time.time()-vrijeme_pocetak:.2f}s")
                vrijeme_pocetak = time.time()
                sum_jabuka = 0
                sum_koraka = 0
        sock.send_pyobj((True, 1))
        model_sub.close()


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Start memory server first
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

    # Shutdown memory server
    client = MemoryClient()
    final_size = client.shutdown()
    mem_proc.join(timeout=2)
    
    print(f"\nAll done. Memory size: {final_size}")









    # trebam promjeniti arhitekturu, trainer 1/2 vremena potroši na komunikaciju sa memorijom