import numpy as np
from typing import Any, NamedTuple


class Experience(NamedTuple):
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool
    gamma: float


class ReplaySampleLog(NamedTuple):
    experience_age: np.ndarray
    weights: np.ndarray
    priorities: np.ndarray
    replay_size: int


class CircularReplayStorage:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.counter = 0

    def push(self, experience):
        write_idx = self.counter
        overwritten = len(self.memory) >= self.capacity

        if overwritten:
            self.memory[write_idx] = experience
        else:
            self.memory.append(experience)

        self.counter = (self.counter + 1) % self.capacity
        return write_idx, overwritten

    def sample(self, data_idxs):
        return [self.memory[i] for i in data_idxs]

    def __len__(self):
        return len(self.memory)

class SumTree:
    """Binary SumTree za efikasno sampliranje i update prioriteta u O(log N)."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.n_entries = 0
        self.head = 0

    @property
    def total(self):
        return self.tree[0]
    
    
    def leaf_index_from_data_idx(self,data_idx: int) -> int:
        return data_idx + self.capacity - 1

    def data_idx_from_leaf_index(self, leaf_idx: int) -> int:
        return leaf_idx - self.capacity + 1

    def add(self, p, data_idx):
        """Dodaj novi sample s prioritetom p."""
        if(self.head != data_idx):
            raise Exception(f"Head mismatch in Sumtree: expected {self.head}, got {data_idx}")
        
        self.update(data_idx, p)
        self.n_entries = min(self.n_entries + 1, self.capacity)
        
        self.head = (self.head + 1)%self.capacity

    def update(self, data_idx, p):
        """Ažuriraj prioritet na indeksu i propagiraj promjenu prema gore."""
        idx = self.leaf_index_from_data_idx(data_idx)
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        if self.n_entries == 0 or self.total <= 0:
            raise ValueError("Cannot sample from an empty SumTree.")
        s = min(max(s, 0.0), self.total - 1e-8)
        leaf_idx = 0
        while leaf_idx < self.capacity - 1:
            left, right = 2 * leaf_idx + 1, 2 * leaf_idx + 2
            if s <= self.tree[left]:
                leaf_idx = left
            else:
                s -= self.tree[left]
                leaf_idx = right
        data_idx = self.data_idx_from_leaf_index(leaf_idx)  

        if data_idx >= self.n_entries:
            data_idx = self.n_entries - 1
            leaf_idx = self.leaf_index_from_data_idx(data_idx)
        return leaf_idx, self.tree[leaf_idx], data_idx
    
    def sample(self,batch_size, beta = 0): # sampla batch_size random brojeva s vjerojatnostima iz priorites 
        cum_sum = self.tree[0]
        s = np.random.uniform(0, cum_sum, size=batch_size)
        data_idxs = []
        probs = np.zeros((batch_size,))
        sample_priorities = np.zeros((batch_size,))
        for i in range(batch_size):
            _, prob, data_idx = self.get(s[i]) 
            data_idxs.append(data_idx)
            probs[i] = prob/cum_sum
            sample_priorities[i] = prob
        probs = np.maximum(probs, 1e-8)
        weights = (1/(self.n_entries * probs)) ** beta
        if weights.size:
            weights /= weights.max()

        sample_log = ReplaySampleLog((self.head-np.array(data_idxs))%self.capacity, weights, sample_priorities, self.n_entries)
        return data_idxs , weights, sample_priorities, sample_log

    def sample_segment(self,batch_size,beta=0): # sampla batch_size random brojeva s vjerojatnostima iz priorites, ali tako da prvo podijeli na segmente jednakih duljina pa iz svakog segmenta izabere smaple ovo je kao malo uniformnije dist priorities 
        cum_sum = self.tree[0]
        segment = cum_sum/batch_size
        data_idxs = []
        probs = np.zeros((batch_size,))
        sample_priorities = np.zeros((batch_size,))
        for i in range(batch_size):
            _, priority, data_idx = self.get(np.random.uniform(segment*i, segment*(i+1)))
            data_idxs.append(data_idx)
            probs[i] = priority/cum_sum
            sample_priorities[i] = priority
        probs = np.maximum(probs, 1e-8)
        weights = (1/(self.n_entries * probs)) ** beta
        if weights.size:
            weights /= weights.max()

        sample_log = ReplaySampleLog((self.head-np.array(data_idxs))%self.capacity, weights, sample_priorities, self.n_entries)
        return data_idxs, weights, sample_priorities, sample_log 
    
    



class BaseReplayBuffer:
    """
    ideja je da je ovo bazna memory klasa, a ti napraviš novu klasu koja nasljeđuje sve od ove, 
    treba promijeniti:
        push tako da se u novoj push funkciji određuje prioritet i zoves super().push(experience, priority)
        update_priorities tako da mijenja prioritet sjecanja kako zelis
    """
    def __init__(self,
                priorities = True, #hoce li biti uniformno distribuirano ili po prioritetima, ako je po prioritetima u push treba slati i prioritet
                weights_bool = True, # hoce li tezine koje vrati program biti sve jedan ili prilagođene s obizrom na distribuciju prioriteta
                predecesor_bool = False, # hoce li se prioriteti propagirat unazad predecesorima
                segment= True, # hoce li se prioriteti smaplirat po segementima(uniformnije) ili cisto po distribuciji prioriteta
                gamma = 0.9, # ovo je gamma koji sluzi samo za propagiranje u nazad predecesorima
                capacity=500_000, 
                n_step_remember = 1, # koliko se koraka u naprijed gleda reward
                alpha_start=0.6, 
                alpha_end = 0.6,            # alpha određuje koliko ce se prioriteti uvažavat  treba biti u intervalu [0,1]: 0 uopće nisu bitni, 1 maksimalno su bitni
                alpha_steps = 1_000_000,    # alpha krece u alpha start i linearno ide prema alpha_end kroz alpha_steps spremanja u memoriju
                beta_start=0.4, 
                beta_end=1.0,               # beta isto kao i alpha samo za znacajnost tezina
                beta_steps=200_000,
                priority_min=1e-6,
                priority_max=10.0,
                ): 
        self.storage = CircularReplayStorage(capacity)
        
        self.predecesor = None
        if predecesor_bool:
            self.predecesor = [False] * capacity
        self.predecesor_bool = predecesor_bool
        
        self.capacity = capacity
        self.n_step = n_step_remember
        self.gamma = gamma

        if priorities:
            self.priorities = SumTree(capacity)
            self.max_priority = 5
        else:
            self.priorities = None
            self.max_priority = 5
        self.mean_td_error = 0.5

        self.segment = segment
        self.alpha = alpha_start
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_steps = alpha_steps
        self.alpha_step_counter = 0

        if weights_bool:
            self.beta = beta_start
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta_steps = beta_steps
        else: 
            self.beta = beta_start
            self.beta_start = 0
            self.beta_end = 0
            self.beta_steps = 10

        self.beta_step_counter = 0
        self.priority_min = priority_min
        self.priority_max = priority_max


        self.num_visits=np.zeros((self.capacity,))
        self.td_errors_log=np.zeros((self.capacity,))
        
    
    def update_beta(self):
        self.beta_step_counter += 1
        fraction = min(1.0, self.beta_step_counter / self.beta_steps)
        self.beta =  self.beta_start + fraction * (self.beta_end - self.beta_start)
    
    def update_alpha(self):
        self.alpha_step_counter += 1
        fraction = min(1.0, self.alpha_step_counter / self.alpha_steps)
        self.alpha = self.alpha_start + fraction * (self.alpha_end - self.alpha_start)

  

    def push(self, experience,priority = None): 
        num_visits = None
        td_error_mean = None
        write_idx = self.storage.counter
        _, overwritten = self.storage.push(experience)

        if not overwritten:
            if self.predecesor_bool:
                if len(self.storage) > self.n_step:
                    self.predecesor[write_idx] = (write_idx - self.n_step) % self.capacity
                else:
                    self.predecesor[write_idx] = False

        else:
            num_visits = self.num_visits[write_idx]
            td_error_mean = self.td_errors_log[write_idx]/ num_visits if num_visits > 0 else 0 
            self.num_visits[write_idx] = 0
            self.td_errors_log[write_idx] = 0
            if self.predecesor_bool:
                self.predecesor[write_idx] = (write_idx - self.n_step) % self.capacity
                self.predecesor[(write_idx + self.n_step) % self.capacity] = False 

        if self.priorities is not None:
                init_p = self.max_priority if (priority is None or np.isnan(priority)) else float(priority)
                init_p = self.clip_priorities([init_p])[0]
                self.priorities.add(init_p**self.alpha,write_idx)

        return (num_visits, td_error_mean) # ovo je za provjeru koliko se puta svako sjecanje iskoristilo i koji je bio prosjecni td error za to sjecanje
    
    def _tree_to_raw_priorities(self, tree_priorities):
        tree_priorities = np.asarray(tree_priorities, dtype=np.float32)
        if self.priorities is None:
            return tree_priorities
        alpha = max(float(self.alpha), 1e-8)
        return np.maximum(tree_priorities, self.priority_min) ** (1.0 / alpha)

    def clip_priorities(self, priorities):
        priorities = np.asarray(priorities, dtype=np.float32)
        priorities = np.nan_to_num(
            priorities,
            nan=self.priority_min,
            posinf=self.priority_max if self.priority_max is not None else self.max_priority,
            neginf=self.priority_min,
        )
        if self.priority_max is None:
            return np.maximum(priorities, self.priority_min)
        return np.clip(priorities, self.priority_min, self.priority_max)

    def priority_policy(self, td, old_priorities):
        return old_priorities

    def apply_priorities(self, data_idxs, td, priorities):
        data_idxs = np.asarray(data_idxs, dtype=np.int64)
        td = np.asarray(td, dtype=np.float32)
        priorities = self.clip_priorities(priorities)

        # logging
        self.num_visits[data_idxs] += 1
        self.td_errors_log[data_idxs] += td
        self.mean_td_error = self.mean_td_error*0.99 + np.mean(td)*0.01 
        
        if priorities.size:
            self.max_priority = max(self.max_priority, float(np.max(priorities)))
        
        if self.priorities is not None:
            for  data_idx, p in zip(data_idxs, priorities):
                p_update= p ** self.alpha
                self.priorities.update(data_idx, p_update)
                
                if self.predecesor_bool is not False and self.predecesor[data_idx] is not False:
                    pred_idx = self.predecesor[data_idx]
                    pred_leaf_idx = self.priorities.leaf_index_from_data_idx(pred_idx)
                    p_pred = min(self.max_priority ** self.alpha, max(self.priorities.tree[pred_leaf_idx], (p * self.gamma**2 )** self.alpha))
                    self.priorities.update(pred_idx, p_pred)

    def update_priorities(self, data_idxs, td, old_priorities):
        priorities = self.priority_policy(td, old_priorities)
        self.apply_priorities(data_idxs, td, priorities)

    def sample(self, batch_size):
        # pretvori prioritete u vjerojatnosti
        if self.priorities == None:
            data_idxs = np.random.choice(len(self.storage), batch_size)
            samples = self.storage.sample(data_idxs)
            weights = np.ones(batch_size, dtype=np.float32)
            old_priorities = np.ones(batch_size, dtype=np.float32)

            sample_log = ReplaySampleLog((self.storage.counter-np.array(data_idxs))%self.capacity, weights, old_priorities, len(self.storage))
        else:
            if self.segment :
                data_idxs, weights, tree_priorities, sample_log = self.priorities.sample_segment(batch_size, self.beta)
            else:
                data_idxs, weights, tree_priorities, sample_log = self.priorities.sample(batch_size, self.beta)

            old_priorities = self._tree_to_raw_priorities(tree_priorities)
            sample_log = ReplaySampleLog(sample_log.experience_age, weights, old_priorities, sample_log.replay_size)


            samples = self.storage.sample(data_idxs)

        self.update_alpha()
        self.update_beta()

        return samples, data_idxs, weights, old_priorities, sample_log
    """
    samples su sjecanja koja se dobiju iz memorije
    data_idxs su indeksi tih sjecanja u memoriji, oni se koriste za update prioriteta
    weights su tezine koje se koriste pri racunanju gubitka
    old_priorities su raw prioriteti tih sjecanja prije updatea; koriste se za priority_policy
    sample_log su dodatne informacije koje mogu biti korisne za logiranje
    to je tuple (broj posjeta, tezine, prioriteti, trenutna velicina memorije)
    """

    def __len__(self):
        return len(self.storage)

    @property
    def memory(self):
        return self.storage.memory

    @property
    def counter(self):
        return self.storage.counter




class TDPriorityReplayBuffer(BaseReplayBuffer):
    """
    memory koji određuje prioritete s obzirom na TD_error koji je bio tijekom učenja iz sjećanja
    """
    def __init__(self, capacity=100_000, gamma=0.93, n_step_remember=1, weights = True, segment= True, predecesor = False,
                  alpha_start=0.5, alpha_end=0.5, alpha_steps=1_000_000, beta_start=0.3, beta_end=1, beta_steps=100_000, eps=1e-6,
                  priority_clip=10.0):
        super().__init__(capacity = capacity,gamma = gamma, n_step_remember = n_step_remember, priorities= True, weights_bool = weights, segment=segment, predecesor_bool= predecesor, 
                        alpha_start=alpha_start, alpha_end=alpha_end, alpha_steps=alpha_steps, beta_end= beta_end, beta_start= beta_start, beta_steps= beta_steps,
                        priority_min=eps, priority_max=priority_clip)
    
        self.eps = eps                  

    def priority_policy(self, td, old_priorities):
        return np.abs(td) + self.eps

    def push(self, experience):
        return super().push(experience, self.max_priority) 


class TDDecayPriorityReplayBuffer(TDPriorityReplayBuffer):
    """
    TD priority s prirodnim padom starih prioriteta.

    Ako je novi TD error velik, sample ostaje bitan. Ako nije, stari prioritet
    pada prema old_priority * priority_decay.
    """
    def __init__(self, *args, priority_decay=0.995, **kwargs):
        super().__init__(*args, **kwargs)
        self.priority_decay = priority_decay

    def priority_policy(self, td, old_priorities):
        td_priority = np.abs(td) + self.eps
        decayed_priority = np.asarray(old_priorities, dtype=np.float32) * self.priority_decay
        return np.maximum(td_priority, decayed_priority)


class TDMixPriorityReplayBuffer(TDPriorityReplayBuffer):
    """
    Meksi blend TD prioriteta i starog prioriteta.

    td_mix=1.0 je standardni TD priority, td_mix=0.0 samo zadrzava stari prioritet.
    """
    def __init__(self, *args, td_mix=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 <= td_mix <= 1.0:
            raise ValueError("td_mix must be between 0.0 and 1.0")
        self.td_mix = td_mix

    def priority_policy(self, td, old_priorities):
        td_priority = np.abs(td) + self.eps
        old_priorities = np.asarray(old_priorities, dtype=np.float32)
        return self.td_mix * td_priority + (1.0 - self.td_mix) * old_priorities


class RewardPriorityReplayBuffer(BaseReplayBuffer):
    """
    Daje veci prioritet sjecanjima s vecim rewardom.
    """
    def __init__(self, capacity=100_000, reward_priority=1, gamma=0.93, n_step_remember=1, weights=True, segment=True, predecesor=False,
                 alpha=0.6, alpha_end=0, alpha_steps=200_000, beta_start=0.4, beta_end=1.0, beta_steps=200_000, eps=1e-6):
        super().__init__(
            capacity=capacity,
            gamma=gamma,
            n_step_remember=n_step_remember,
            priorities=True,
            weights_bool=weights,
            segment=segment,
            predecesor_bool=predecesor,
            alpha_start=alpha,
            alpha_end=alpha_end,
            alpha_steps=alpha_steps,
            beta_end=beta_end,
            beta_start=beta_start,
            beta_steps=beta_steps,
            priority_min=eps,
        )

        self.eps = eps
        self.reward_priority = reward_priority

    def priority_policy(self, td, old_priorities):
        return old_priorities * self.gamma**2

    def push(self, experience):
        reward = experience.reward
        priority = 1 + self.reward_priority * abs(reward)
        return super().push(experience, priority)


class ReplayBuffer(BaseReplayBuffer):
    """
    uniformno prioretizira
    """
    def __init__(self, capacity=100_000, gamma=0.93, n_step_remember=1):
        super().__init__(capacity = capacity, n_step_remember = n_step_remember, priorities=False, weights_bool = False, segment=False, predecesor_bool= False)
    
    def update_priorities(self, indices, td, old_priorities):
        return super().update_priorities(indices,td, old_priorities)


ExperienceMemory = BaseReplayBuffer
