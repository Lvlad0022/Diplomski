
from q_logic import Agent
from simplebot import HeatmapBot
from for_logging import CSVLogger
import os
import random
import time
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory (AIBG-9.0-master/)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to Python's search path
sys.path.insert(0, parent_dir)

print(  )
from logic.game import SnakeGame

def serialize_game_state(game):
    """
    Serializira stanje igre u jednostavan format rječnika (dictionary).

    Args:
        game: Glavni objekt igre koji sadrži puno stanje igre.

    Returns:
        Rječnik (dictionary) koji predstavlja pojednostavljeno stanje igre.
    """
    return {
        'map': game.board.map,
        'players': [
            {
                'name': player.name,
                'score': player.score,
                'body': player.body,
                # Napomena: camelCase iz JS-a (activeItems) je preveden u snake_case (active_items)
                # što je standard u Pythonu.
                'activeItems': player.active_items,
                'lastMoveDirection': player.last_move_direction,
                'nextMoveDirection': player.next_move_direction,
            }
            # List comprehension je Pythonov način za ono što je .map() u JavaScriptu
            for player in game.players
        ],
        'winner': game.winner,
        'moveCount': game.move_count,
    }

def granica(count, avg_count):
    return 0.2+ (count/avg_count)*0.8




# tu ide logika za igrace 
id1 = "id1" 
id2 = "id2"
name1 = "ime1"
name2 = "ime2"
CHECKPOINT_FOLDER = 'model_checkpoints'

# Kreiraj taj folder ako ne postoji
if not os.path.exists(CHECKPOINT_FOLDER):
    os.makedirs(CHECKPOINT_FOLDER)

# Definiraj relativne putanje za spremanje koristeći os.path.join
# Ovo će stvoriti putanje poput 'model_checkpoints/save_agent.pth'
for i in [3,5,7]:
    file_path = os.path.join(CHECKPOINT_FOLDER, 'save_agent.pth')
    log_path = os.path.join(CHECKPOINT_FOLDER, 'save_agent.pth')
    file_path_simple = os.path.join(CHECKPOINT_FOLDER, f'save_simple_resnet_n{i}.pth')
    log_path_simple = os.path.join(CHECKPOINT_FOLDER, f'save_simle_resnet_log_n{i}.csv')
    agent1 = Agent(name1,n_step_remember=i, gamma=0.80)
    agent1.save_agent_state(file_path_simple)
    agent2 = Agent(name2)

    simplebot = HeatmapBot(name2)

    num_games = 7000

    logger = CSVLogger(log_path_simple, fieldnames=[
        "game", "vrijeme1", "vrijeme2", "sum_reward", "avg_count",
        "win_pct", "loss_moving_avg", "epsilon", "learning_rate", "num_moves"
    ])
    avg_count = 0
    moving_avg = 0
    win_pct = 0
    loss_moving_avg= 0
    for i in range(num_games):
        game = SnakeGame()
        game.add_player({"id": id1, "name": name1})
        game.add_player({"id": id2, "name": name2})


        GameOver = False

        data = serialize_game_state(game)
        sum_reward = 0

        count = 0
        avg_loss = 0
        sum_loss = 0
        
        vrijeme1 = 0
        vrijeme2 = 0


        while not GameOver:
            
            moves = []
            a = time.time()
            moves.append({"playerId": id1 , "direction": agent1.get_action(data)})
            b = time.time()
            
            moves.append({"playerId": id2 , "direction": simplebot.get_action(data)})
            c= time.time()

            vrijeme1 += (b-a)
            vrijeme2 += (c-b)
            game.process_moves(moves)

            novi_data = serialize_game_state(game)
            reward = agent1.give_reward(novi_data,data)
            sum_reward += reward

            p = random.randint(0,1)
            if(i < 4000 or p < granica(count,avg_count) ):
                agent1.remember(data,novi_data)

            if(count%8 == 0 and i >5):
                count+=1
                sum_loss += agent1.train_long_term()

            data = novi_data
            GameOver = game.check_game_over()
            if(GameOver):
                if novi_data.get('winner') == name1:
                    win_pct = win_pct * 0.99 + 0.01
                else:
                    win_pct = win_pct * 0.99 

        #sum_loss += agent1.train_long_term()

        avg_count = (count)*0.01 + avg_count*0.99
        moving_avg = (sum_reward/count)*0.01 + moving_avg*0.99
        loss_moving_avg = (sum_loss/count)*0.01 + loss_moving_avg*0.99
        if((i+1)%500==0 and i >100):
            agent1.save_agent_state(file_path_simple)
        
        n_games, epsilon, lr = agent1.get_model_state()

        print(f"igra: {i}")
        print(f"vrijeme1: {vrijeme1/count}")
        print(f"vrijeme2: {vrijeme2/count}")
        print(f"sum_reward: {moving_avg}")
        print(f"avg count:{avg_count}")
        print(f"win_pct: {win_pct}")
        print(f"loss moving_avg: {loss_moving_avg}")
        print(f"epsilon: {epsilon}")
        print(f"learning rate: {lr}")
        print(f"broj_poteza{count}")
        logger.log({
            "game": i,
            "vrijeme1": vrijeme1 / count,
            "vrijeme2": vrijeme2 / count,
            "sum_reward": moving_avg,
            "avg_count": avg_count,
            "win_pct": win_pct,
            "loss_moving_avg": loss_moving_avg,
            "epsilon": epsilon,
            "learning_rate": lr,
            "num_moves": count,
        })


    if(num_games):
        agent1.save_agent_state(file_path_simple)

agent2.load_agent_state(file_path_simple, training=False)
num_games = 0
change_every =0

avg_count = 0
win_pct = 0.5
loss_moving_avg = 0
for i in range(num_games):
    game = SnakeGame()
    game.add_player({"id": id1, "name": name1})
    game.add_player({"id": id2, "name": name2})

    GameOver = False

    data = serialize_game_state(game)
    count = 0
    sum_reward = 0
    
    
    sum_loss = 0
    vrijeme1 = 0
    vrijeme2 = 0
    while not GameOver:
        count +=1
        moves = []
        a = time.time()
        moves.append({"playerId": id1 , "direction": agent1.get_action(data)})
        b = time.time()
        
        moves.append({"playerId": id2 , "direction": agent2.get_action(data)})
        c= time.time()

        vrijeme1 += (b-a)
        vrijeme2 += (c-b)
        game.process_moves(moves)

        
        novi_data = serialize_game_state(game)
        reward = agent1.give_reward(novi_data,data)
        sum_reward += reward

        p = random.randint(0,1)
        if(i < 4000 or p < granica(count, avg_count) ):
            agent1.remember(data,novi_data)
        if(count%4 == 0 and i > 20):
            sum_loss += agent1.train_long_term()

        data = novi_data
        GameOver = game.check_game_over()
        if GameOver: # Provjeri je li igra gotova
            if novi_data.get('winner') == name1:
                win_pct = win_pct * 0.95 + 0.05
            else:
                win_pct = win_pct * 0.95
        

    #sum_loss += agent1.train_long_term()

    loss_moving_avg = loss_moving_avg*0.95+ (sum_loss/ (count//32+1))*0.05
    moving_avg = (sum_reward/count)*0.05 + moving_avg*0.95
    avg_count = (count)*0.05 + avg_count*0.95
    if (i+1)%change_every == 0:
        agent1.save_agent_state(file_path)
        agent2.change_weights(agent1)
    print(f"igra: {i}")
    print(f"vrijeme1: {vrijeme1/count}")
    print(f"vrijeme2: {vrijeme2/count}")
    print(f"sum_reward: {sum_reward}")
    print(f"avg_reward: {moving_avg}")
    print(f"avg count:{avg_count}")
    print(f"win_pct: {win_pct}")
    print(f"loss moving_avg: {loss_moving_avg}")
    print(f"broj_poteza{count}")

agent1.save_agent_state(file_path)
print("kraj")

    