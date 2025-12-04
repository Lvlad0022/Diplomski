from q_logic_snake import SimpleSnakeAgent
from simplebot import HeatmapBot
from q_logic.q_logic_logging import make_run_name, CSVLogger
import os
import random
import time
import sys
from Diplomski.q_logic.q_logic import set_seed



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





# tu ide logika za igrace 
id1 = "id1" 
id2 = "id2"
name1 = "ime1"
name2 = "ime2"
CHECKPOINT_FOLDER = 'simple_snake_game_models'
set_seed(42)

# Kreiraj taj folder ako ne postoji
if not os.path.exists(CHECKPOINT_FOLDER):
    os.makedirs(CHECKPOINT_FOLDER)

# Definiraj relativne putanje za spremanje koristeći os.path.join
# Ovo će stvoriti putanje poput 'model_checkpoints/save_agent.pth'
for snake_i in ["dueling"]:
    for scheduler_i in [1]:
        for memory_i in [1]:
            for b in[0.85]:
                file_name = f'zsave_simple_snake3_log_n3_{snake_i}_memory{memory_i}_scheduler{scheduler_i}_gamma{b}'
                file_name_date = make_run_name(file_name)
                log_path_simple = os.path.join(CHECKPOINT_FOLDER, f'{file_name_date}.csv')

                agent1 = SimpleSnakeAgent(player1_name = name1, n_step_remember=3, gamma=b, memory= memory_i,snake_i=snake_i, scheduler=scheduler_i)
                                        # advanced_logging_path= file_name, time_logging_path = file_name)

                simplebot = HeatmapBot(name2)



                num_games = 10000

                logger = CSVLogger(log_path_simple, fieldnames=[
                    "game", "vrijeme_izvodenja", "vrijeme_treniranja", "sum_reward", "avg_count", "win_pct",
                    "loss_moving_avg", "epsilon", "learning_rate", "num_moves"    ])
                avg_count = 0
                moving_avg = 0
                win_pct = 0
                loss_moving_avg= 0
                for i in range(num_games):
                    time_pocetak = time.time()
                    
                    game = SnakeGame()
                    game.add_player({"id": id1, "name": name1})
                    game.add_player({"id": id2, "name": name2})

                    GameOver = False

                    data = serialize_game_state(game)
                    sum_reward = 0

                    avg_loss = 0
                    sum_loss = 0
                    
                    vrijeme1 = 0
                    vrijeme2 = 0
                    vrijeme3 = 0
                    vrijeme4 = 0


                    sum_count = 0        
                    count = 0
                    while not GameOver:
                        
                        count+= 1
                        moves = []
                        a = time.time()
                        moves.append({"playerId": id1 , "direction": agent1.get_action(data)})
                        b = time.time()
                        
                        moves.append({"playerId": id2 , "direction": simplebot.get_action(data)})
                        c= time.time()

                        vrijeme1 += (b-a)
                        vrijeme2 += (c-b)

                        b = time.time()
                        game.process_moves(moves)
                        c= time.time()
                        vrijeme4 += (c-b)


                        novi_data = serialize_game_state(game)
                        reward, done= agent1.give_reward(novi_data,data,1)
                        sum_reward += reward

                        agent1.remember(data,novi_data)

                        
                        GameOver = game.check_game_over()
                        if(GameOver):
                            if novi_data.get('winner') == name1:
                                win_pct = win_pct * 0.99 + 0.01
                            else:
                                win_pct = win_pct * 0.99 

                        if( (i >100 and count%8== 0) or i>1000):
                            sum_count += 1
                            a = time.time()
                            loss = agent1.train_long_term()
                            sum_loss += loss
                            vrijeme3 += time.time() - a
                        
                        data = novi_data

                    #sum_loss += agent1.train_long_term()

                    model_target_update_counter = agent1.trainer.model_target_update_counter
                    avg_count = (count)*0.01 + avg_count*0.99
                    moving_avg = (sum_reward/count)*0.01 + moving_avg*0.99
                    loss_moving_avg = (sum_loss / (sum_count if sum_count else 1)) * 0.01 + loss_moving_avg*0.99
                    vrijeme_treniranja = vrijeme3/(sum_count if sum_count else 1)

                    if(i%2000==0 and i >100):
                       agent1.save_agent_state(file_name_date)
                
                    n_games, epsilon, lr = agent1.get_model_state()

                    cijeli_ciklus = time.time()- time_pocetak

                    print(f"igra: {i}")
                    print(f"cijeli_ciklus: {cijeli_ciklus}")
                    print(f"vrijeme po potezu: {cijeli_ciklus/count}")
                    print(f"vrijeme izvođenja: {vrijeme1/count}")
                    print(f"vrijeme bota: {vrijeme2/count}")
                    print(f"vrijeme treniranja: {vrijeme_treniranja}")
                    print(f"vrijeme procesuiranja igre: {vrijeme4/count}")
                    print(f"sum_reward: {moving_avg}")
                    print(f"avg count:{avg_count}")
                    print(f"loss moving_avg: {loss_moving_avg}")
                    print(f"epsilon: {epsilon}")
                    print(f"learning rate: {lr}")
                    print(f"win_pct {win_pct}")
                    print(f"broj_poteza{count}")
                    logger.log({
                        "game": i,
                        "vrijeme_izvodenja": vrijeme1/count,
                        "vrijeme_treniranja": vrijeme_treniranja,
                        "sum_reward": moving_avg,
                        "avg_count": avg_count,
                        "win_pct": win_pct,
                        "loss_moving_avg": loss_moving_avg,
                        "epsilon": epsilon,
                        "learning_rate": lr,
                        "num_moves": count,
                    })
   # if(num_games):
    #    agent1.save_agent_state(file_name)

'''
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

'''