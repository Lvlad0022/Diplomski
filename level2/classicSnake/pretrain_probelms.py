import numpy as np
from collections import deque

"""
problem definirani za predtreniranje backbone mreže
problem1 # koordinate glave
problem2 # duljina tijela
problem3 # udaljenost glave do prve prepreke u bilo kojem smjeru
problem4 # je li potez siguran
problem5 # udaljenost do zida u svim smjerovima
problem6 # lokacija repa
problem7 # koordinate jabuke
problem8 # može li se do jabuke sigurno doći
problem9 # broj okupiranih polja
gt_tail_future # gdej će rep biti za k poteza
"""

def get_coords(mask):
    (ys, xs) = np.where(mask == 1)
    if len(ys) == 0:
        return None
    return np.array([xs[0],ys[0]])

def problem1(x): # koordinate glave
    head = x[0]
    return get_coords(head)

def gt_head_coordinates(x):
    return problem1(x)


def problem2(x): # duljina tijela
    body = x[1]
    return np.array([body.sum()])

def problem3(x): # udaljenost glave do prve prepreke u bilo kojem smjeru
    H, W = x.shape[1], x.shape[2]
    head_y, head_x = problem1(x)
    body = x[1]

    directions = [
        (-1, 0),  # N
        (-1, 1),  # NE
        (0, 1),   # E
        (1, 1),   # SE
        (1, 0),   # S
        (1, -1),  # SW
        (0, -1),  # W
        (-1, -1)  # NW
    ]

    dists = []
    for dy, dx in directions:
        y, x = head_y, head_x
        dist = 0
        while True:
            y += dy
            x += dx
            if y < 0 or y >= H or x < 0 or x >= W:
                break
            if body[y, x] == 1:
                break
            dist += 1
        dists.append(dist)

    return np.array(dists, dtype=np.int32)



def problem4(x): # je li potez siguran
    H, W = x.shape[1], x.shape[2]
    head_y, head_x = problem1(x)
    body = x[1]

    moves = {
        "up":    (-1, 0),
        "right": (0, 1),
        "left":  (0, -1),
        "down":  (1, 0)
    }

    safe = []
    for dy, dx in moves.values():
        y = head_y + dy
        x = head_x + dx
        if y < 0 or y >= H or x < 0 or x >= W:
            safe.append(0)
        elif body[y, x] == 1:
            safe.append(0)
        else:
            safe.append(1)

    return np.array(safe, dtype=np.int32) 

def problem5(x): # udaljenost do zida u svim smjerovima
    H, W = x.shape[1], x.shape[2]
    y, x = problem1(x)

    return np.array([
        y,              
        W - 1 - x,      
        H - 1 - y,      
        x               
    ], dtype=np.int32)


def problem6(x): #lokacija repa
    decay = x[3]
    decay[decay == 0] = 10
    if decay.min() == 10:
        return np.array((-1, -1))
    ys, xs = np.where(decay == decay.min())
    return np.array([ys[0], xs[0]])


def gt_tail_future(x, k, map_size= 100): #gdje će biti rep za k poteza
    decay = x[3]
    if decay.max() == 0:
        return np.array((-1, -1))

    decay[decay == 0] = 10

    tail_val = decay.min()
    target_val = tail_val + k / map_size

    ys, xs = np.where((decay- target_val) < 1e-8)
    if len(ys) == 0:
        return np.array((-1, -1))
    return np.array((ys[0], xs[0]))




def problem7(x): # koordinate jabuke
    #applecoord
    return get_coords(x[2])

def gt_apple_coordinates(x):
    return problem7(x)


def problem8(x): #može li se do jabuke sigurno doći
    H, W = x.shape[1], x.shape[2]
    body = x[1]
    head = problem1(x)
    apple = gt_apple_coordinates(x)

    if apple is None:
        return 0

    # Remove current tail because it will move
    tail = gt_tail_plus_1 = gt_tail_future(x, 1)
    if tail[0] != -1 :
        ty, tx = tail
        body[ty, tx] = 0

    q = deque([head])
    visited = set([head])

    while q:
        y, x = q.popleft()
        if (y, x) == apple:
            return np.array((1))

        for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W:
                if body[ny, nx] == 1:
                    continue
                if (ny, nx) not in visited:
                    visited.add((ny, nx))
                    q.append((ny, nx))

    return np.array((0))



def problem9(x): #broj okupiranih polja
    occupied = (x[0] + x[1] + x[2]) > 0
    return np.array([(occupied == 0).sum()])




###################################################### kasnije
def map_problem1(x):
    occupied = (x[0] + x[1] + x[2]) > 0
    return (occupied == 0).astype(np.int32)

def map_problem2(x):
    return ((x[0] + x[1]) > 0).astype(np.int32)
