import numpy as np

class SumTree:
    """Binary SumTree za efikasno sampliranje i update prioriteta u O(log N)."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.n_entries = 0

    @property
    def total(self):
        return self.tree[0]
    
    
    def leaf_index_from_data_idx(self,data_idx: int) -> int:
        return data_idx + self.capacity - 1

    def data_idx_from_leaf_index(self, leaf_idx: int) -> int:
        return leaf_idx - self.capacity + 1

    def add(self, p, data_idx):
        """Dodaj novi sample s prioritetom p."""
        self.update(data_idx, p)
        self.n_entries = min(self.n_entries + 1, self.capacity)

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
        idxs = []
        probs = np.zeros((batch_size,))
        for i in range(batch_size):
            _, prob, data_idx = self.get(s[i]) 
            idxs.append(data_idx)
            probs[i] = prob/cum_sum
        weights = (1/(self.n_entries * probs)) ** beta
        if weights.size:
            weights /= weights.max()
        return idxs , weights

    def sample_segment(self,batch_size,beta=0): # sampla batch_size random brojeva s vjerojatnostima iz priorites, ali tako da prvo podijeli na segmente jednakih duljina pa iz svakog segmenta izabere smaple ovo je kao malo uniformnije dist priorities 
        cum_sum = self.tree[0]
        segment = cum_sum/batch_size
        data_idxs = []
        probs = np.zeros((batch_size,))
        for i in range(batch_size):
            _, prob, data_idx = self.get(np.random.uniform(segment*i, segment*(i+1)))
            data_idxs.append(data_idx)
            probs[i] = prob/cum_sum
        weights = (1/(self.n_entries * probs)) ** beta
        if weights.size:
            weights /= weights.max()
        return data_idxs, weights


class ExperienceMemory:
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
                capacity=100_000, 
                n_step_remember = 1, # koliko se koraka u naprijed gleda reward
                alpha_start=0.6, 
                alpha_end = 0.6,            # alpha određuje koliko ce se prioriteti uvažavat  treba biti u intervalu [0,1]: 0 uopće nisu bitni, 1 maksimalno su bitni
                alpha_steps = 1_000_000,    # alpha krece u alpha start i linearno ide prema alpha_end kroz alpha_steps spremanja u memoriju
                beta_start=0.4, 
                beta_end=1.0,               # beta isto kao i alpha samo za znacajnost tezina
                beta_steps=200_000
                ): 
        self.memory = []
        self.predecesor = [False] * capacity
        self.predecesor_bool = predecesor_bool
        
        self.capacity = capacity
        self.counter = 0
        self.n_step = n_step_remember
        self.gamma = gamma

        if priorities:
            self.priorities = SumTree(capacity)
            self.max_priority = 5
        else:
            self.priorities = None
            self.max_priority = 5

        self.segment = segment
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_steps = alpha_steps
        self.alpha_step_counter = 0

        if weights_bool:
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta_steps = beta_steps
        else: 
            self.beta_start = 0
            self.beta_end = 0
            self.beta_steps = 10

        self.beta_step_counter = 0
        
    
    def beta(self):
        fraction = min(1.0, self.beta_step_counter / self.beta_steps)
        self.beta_step_counter += 1
        return self.beta_start + fraction * (self.beta_end - self.beta_start)
    
    
    def alpha(self):
        fraction = min(1.0, self.alpha_step_counter / self.alpha_steps)
        self.alpha_step_counter += 1
        return self.alpha_start + fraction * (self.alpha_end - self.alpha_start)

  

    def push(self, experience,priority = None): 
        if len(self.memory) < self.capacity:
            self.memory.append(experience)

            if self.predecesor_bool:
                if len(self.memory) > self.n_step:
                    self.predecesor[self.counter] = (self.counter - self.n_step) % self.capacity
                else:
                    self.predecesor[self.counter] = False
            
            
        else:
            self.memory[self.counter] = experience
            if self.predecesor_bool:
                self.predecesor[self.counter] = (self.counter - self.n_step) % self.capacity
                self.predecesor[(self.counter + self.n_step) % self.capacity] = False 

        if self.priorities is not None:
                init_p = self.max_priority if (priority is None or np.isnan(priority)) else float(priority)
                self.priorities.add(init_p**self.alpha(),self.counter)

        self.counter = (self.counter+1) % self.capacity
    
    def update_priorities(self, data_idxs, priority):
        if self.priorities is not None:
            for  data_idx, p in zip(data_idxs, priority):
                self.priorities.update(data_idx, min(p ** self.alpha(), self.max_priority))

                pred_idx = self.predecesor[data_idx]
                if  pred_idx is not False:
                    self.priorities.update(pred_idx, min(self.max_priority, max( self.priorities.tree[pred_idx], (p * self.gamma**2 )** self.alpha())))

    def sample(self, batch_size):
        # pretvori prioritete u vjerojatnosti
        if self.priorities == None:
            data_idxs = np.random.choice(len(self.memory ), batch_size)
            samples = [self.memory[i] for i in data_idxs]
            weights = np.ones(batch_size, dtype=np.float32)
        else:
            if self.segment :
                data_idxs, weights = self.priorities.sample_segment(batch_size, self.beta())
            else:
                data_idxs, weights = self.priorities.sample(batch_size, self.beta())


            samples = [self.memory[i] for i in data_idxs] 


        return samples, data_idxs, weights

    def __len__(self):
        return len(self.memory)




class TDPriorityReplayBuffer(ExperienceMemory):
    """
    memory koji određuje prioritete s obzirom na TD_error koji je bio tijekom učenja iz sjećanja
    """
    def __init__(self, capacity=100_000, gamma=0.93, n_step_remember=1, weights = True, segment= True, predecesor = False,
                  alpha_start=0.6, alpha_end=0.6, alpha_steps=1_000_000, beta_start=0.4, beta_end=1.0, beta_steps=200_000, eps=1e-6):
        super().__init__(capacity = capacity,gamma = gamma, n_step_remember = n_step_remember, priorities= True, weights_bool = weights, segment=segment, predecesor_bool= predecesor, 
                        alpha_start=alpha_start, alpha_end=alpha_end, alpha_steps=alpha_steps, beta_end= beta_end, beta_start= beta_start, beta_steps= beta_steps )
    
        self.eps = eps                  

    def update_priorities(self, indices, td_errors):
        priorities = np.abs(td_errors) + self.eps
        super().update_priorities(indices,priorities)

    def push(self, experience):
        super().push(experience, self.max_priority) 


class RewardPriorityReplayBuffer(ExperienceMemory):
    """
    daje veći prioritet sjecanjima s vecim rewardom jer racuna da su ona bitnija
    """
    def __init__(self, capacity=100_000, reward_priority= 1, gamma=0.93, n_step_remember=1, weights = True, segment= True, predecesor = False,
                 alpha=0.6,alpha_end=0.1, alpha_steps = 1_000_000, beta_start=0.4, beta_end=1.0, beta_steps=200_000, eps=1e-6):
        super().__init__(capacity = capacity, gamma = gamma,n_step_remember= n_step_remember, priorities=True, weights_bool = weights, segment=segment, predecesor_bool= predecesor,
                         alpha_start= alpha, alpha_end = alpha_end, alpha_steps = alpha_steps, beta_end= beta_end, beta_start= beta_start, beta_steps= beta_steps )
    
        self.eps = eps                  
        self.reward_priority = reward_priority

    def update_priorities(self, indices, _ ):
        priorities = priorities* self.gamma**2
        super().update_priorities(indices,priorities)

    def push(self, experience):
        reward = experience[3]
        p = 1+ self.reward_priority*reward
        super().push(experience, p)


class ReplayBuffer(ExperienceMemory):
    """
    uniformno prioretizira
    """
    def __init__(self, capacity=100_000, gamma=0.93, n_step_remember=1):
        super().__init__(capacity = capacity, n_step_remember = n_step_remember, priorities=False, weights_bool = False, segment=False, predecesor_bool= False)
    
    
                     
