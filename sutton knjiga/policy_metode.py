import numpy as np 

INFINITY = 10000000000

def policy_value_estimation_in_place(states,policy,next_state, eps = 0.9, termination_eps = 0.001, terminal_states = []): 
    #states je lista svih mogućih stanja, 
    #policy rijecnik koji prima stanje i daje ti  listu tuplea stanja u koje može preći i vjerojatnost prelaska
    # rewards prima dva stanja te nagradu prelaska iz jednog u drugo stanje

    n = len(states)

    values = np.zeros(n)

    state_to_index ={state: idx for idx, state in enumerate(states)}

    num_iterations= 0

    running = True
    while running:
        num_iterations +=1
        running = False  
        for state1 in states:
            a = 0
            if state1 in terminal_states:
                values[state_to_index[state1]] = 0
                continue
            for move,prob1 in policy[state1]:
                for state2,reward,prob2 in next_state[state1,move]:
                    a += prob1*prob2*(reward + eps*values[state_to_index[state2]])
            if(abs(values[state_to_index[state1]]-a) > termination_eps):
                running = True
            values[state_to_index[state1]] = a
    
    return values, num_iterations


def policy_value_estimation(states,policy,next_state, eps = 0.9, termination_eps = 0.001,terminal_states = []): 
    #states je lista svih mogućih stanja, 
    #policy prima stanje i daje ti stanja u koje može preći i vjerojatnost prelaska
    # rewards prima dva stanja te nagradu prelaska iz jednog u drugo stanje

    n = len(states)

    values = np.zeros(n)
    values2 = np.zeros(n)
    state_to_index ={state: idx for idx, state in enumerate(states)}
    num_iterations= 0

    running = True
    while running:
        num_iterations +=1
        running = False  
        for state1 in states:
            if state1 in terminal_states:
                values[state_to_index[state1]] = 0
                continue
            a = 0
            for move,prob1 in policy[state1]:
                for state2,reward,prob2 in next_state[state1,move]:
                    a += prob1*prob2*(reward + eps*values[state_to_index[state2]])
            if(abs(values[state_to_index[state1]]-a) > termination_eps):
                running = True
            values2[state_to_index[state1]] = a
        values = values2.copy()
        
        
    
    return values, num_iterations

def policy_from_value(states,moves,next_state,values,eps = 0.9,terminal_states = []):
    policy = {}

    state_to_index ={state: idx for idx, state in enumerate(states)}

    for state1 in states:
            a = 0
            max_value = -INFINITY
            max_values_move = None
            for move in moves[state1]:
                for state2,reward,prob in next_state[state1,move]:
                    a += prob*(reward + eps*values[state_to_index[state2]])
                if(a > max_value):
                    max_value = a
                    max_values_move = move
                policy[state1] = [(max_values_move,1)]
    return policy

def generalized_policy_iteration(states,policy,next_state,moves, eps = 0.9, termination_eps = 0.001,terminal_states = []):
    
    n = len(states)
    values = np.zeros(n)

    policy
    running = True
    while(running):
        running = False
        values1, it = policy_value_estimation_in_place(states,policy,next_state,terminal_states = terminal_states)
        for  i in range(n):
            if(abs(values[i]-values1[i]) > eps):
                running = True
                break

        values = values1.copy()
        policy = policy_from_value(states,moves,next_state,values)


def value_iteration(states,policy,next_state,moves, eps = 0.9, termination_eps = 0.001,terminal_states = []): 
    #states je lista svih mogućih stanja, 
    #policy prima stanje i daje ti stanja u koje može preći i vjerojatnost prelaska
    # rewards prima dva stanja te nagradu prelaska iz jednog u drugo stanje

    n = len(states)

    values = np.zeros(n)

    num_iterations= 0
    state_to_index ={state: idx for idx, state in enumerate(states)}

    running = True
    while running:
        num_iterations +=1
        running = False  
        for state1 in states:
            if state1 in terminal_states:
                values[state_to_index[state1]] = 0
                continue
            a = 0
            
            for move,prob1 in policy[state1]:
                for state2,reward,prob2 in next_state[state1,move]:
                    a += prob1*prob2*(reward + eps*values[state_to_index[state2]])
            if(abs(values[state_to_index[state1]]-a) > termination_eps):
                running = True
            values[state_to_index[state1]] = a
        
        policy = policy_from_value(states,moves,next_state,values)

    return values, policy,num_iterations



#primjeri igara
# jack dealrship
'''
states = [(i,j) for i in range(21) for j in range(21)]
moves = {}
brojac = 0
for state in states:
    m,n = state
    
    moves[state] = [(i,0) for i in range(max(5,m))] + [(0,j) for j in range(max(5,n))]


lambda1_req = 3
lambda2_req = 4
lambda1_ret = 3
lambda2_ret = 2

def exp_prob(lam,x):
    return (lam**x)*np.exp(-lam)/np.math.factorial(x)

next_state = {}
for state in states:
    for move in moves[state]:
        next_state[state,move] = []
        m,n = state
        m_d,n_d = move
        m2 , n2 =m-m_d+n_d, n-n_d+m_d
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    for l in range(5):
                        m3 = min(max(0,m2-i+k),20)
                        n3 = min(max(0,n2-j+l),20)
                        reward = 10*(min(m2,i) + min(n2,j)) 
                        prob = exp_prob(lambda1_ret,k)*exp_prob(lambda2_ret,l)*exp_prob(lambda1_req,i)*exp_prob(lambda2_req,j)
                        next_state[state,move].append(((m3,n3),reward,prob))


policy1 ={}
policy2 = {}

for state in states:
    policy1[state] = [(moves[state][0],1)]
    policy2[state] = [(move, 1/len(moves[state])) for move in moves[state]]

a,b =   policy_value_estimation_in_place(states,policy1,next_state)

'''

###gridworld


n = 4

def in_borders(x,y):
    return 0 <= x < n and 0 <= y < n


states = []
terminal_states = [(n-1,n-1),(0,0)]

for i in range(n):
    for j in range(n):
        states.append((i,j))

moves = {state: [(-1,0),(1,0),(0,1),(0,-1)] for state in states}

next_state = {}

for state in states:
    for move in moves[state]:
        x,y = state
        xd,yd = move
        x1,y1 = x+xd,y+yd
        reward = -1
        if(in_borders(x1,y1)):
            next_state[state,move] = [((x1,y1),-1,1)]
        else:
            next_state[state,move] = [(state,-1,1)]

policy1 ={}
policy2 = {}

for state in states:
    policy1[state] = [(moves[state][0],1)]
    policy2[state] = [(move, 1/len(moves[state])) for move in moves[state]]

values,b =   policy_value_estimation(states,policy2,next_state, eps = 1,terminal_states= terminal_states)

state_to_index ={state: idx for idx, state in enumerate(states)}

for i in range(n):

    for j in range(n):
        print(f"{(values[state_to_index[(i,j)]]):3}",end="")
    print()