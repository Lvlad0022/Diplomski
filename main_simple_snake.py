
from q_logic_simple_snake import SimpleSnakeAgent , SimpleSnakeAgent3 
from q_logic_univerzalno import set_seed
from simplebot import HeatmapBot
from q_logic_logging import make_run_name, CSVLogger
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
from simple_snake_logic.game import SnakeGame



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
                'name': game.player.name,
                'score': game.player.score,
                'body': game.player.body,
                # Napomena: camelCase iz JS-a (activeItems) je preveden u snake_case (active_items)
                # što je standard u Pythonu.
                'activeItems': game.player.active_items,
                'lastMoveDirection': game.player.last_move_direction,
                'nextMoveDirection': game.player.next_move_direction,
            }
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
CHECKPOINT_FOLDER = 'simple_snake_game_models'
set_seed(42)

# Kreiraj taj folder ako ne postoji
if not os.path.exists(CHECKPOINT_FOLDER):
    os.makedirs(CHECKPOINT_FOLDER)

# Definiraj relativne putanje za spremanje koristeći os.path.join
# Ovo će stvoriti putanje poput 'model_checkpoints/save_agent.pth'
for snake_i in ["dueling"]:
    for scheduler_i in [3,4]:
        for memory_i in [0,6]:
            for b in[0.85]:
                file_path = os.path.join(CHECKPOINT_FOLDER, 'save_agent.pth')
                log_path = os.path.join(CHECKPOINT_FOLDER, 'save_agent.pth')
                file_name = f'zsave_simple_snake3_log_n3_{snake_i}_memory{memory_i}_scheduler{scheduler_i}_gamma{b}'
                log_path_simple = os.path.join(CHECKPOINT_FOLDER, f'{make_run_name(file_name)}.csv')

                agent1 = SimpleSnakeAgent3(n_step_remember=3, gamma=b, memory= memory_i,snake_i=snake_i, scheduler=scheduler_i,
                                         advanced_logging_path= file_name, time_logging_path = file_name)#advanced_logging_path= file_name
                #agent1.load_agent_state(str = r"C:\Users\lovro\Desktop\AIBG-9.0-master\simple_snake_game_models\pretrained_net.pth")

                num_games = 8000

                logger = CSVLogger(log_path_simple, fieldnames=[
                    "game", "vrijeme_izvodenja", "vrijeme_treniranja", "sum_reward", "avg_count",
                    "loss_moving_avg", "epsilon", "learning_rate", "num_moves"    ])
                avg_count = 0
                moving_avg = 0
                win_pct = 0
                loss_moving_avg= 0
                for i in range(num_games):
                    if i==5000:
                        fhf = 1

                    game = SnakeGame()
                    game.add_player({"id": id1, "name": name1})


                    GameOver = False

                    data = serialize_game_state(game)
                    sum_reward = 0

                    avg_loss = 0
                    sum_loss = 0
                    
                    vrijeme1 = 0
                    vrijeme2 = 0


                    sum_count = 0        
                    count = 0
                    while not GameOver:
                        count+= 1
                        a = time.time()
                        moves = {"playerId": id1 , "direction": agent1.get_action(data)}
                        vrijeme1 += time.time() - a
                        game.process_moves(moves)

                        novi_data = serialize_game_state(game)
                        reward,done = agent1.give_reward(novi_data, data, None)
                        sum_reward += reward

                        p = random.randint(0,1)
                        if(i < 4000 or p < granica(count,avg_count) ):
                            agent1.remember(data,novi_data)

                        if(count%8 == 0 and i >5):
                            sum_count += 1
                            a = time.time()
                            loss = agent1.train_long_term()
                            sum_loss += loss
                            vrijeme2 += time.time() - a

                        data = novi_data
                        GameOver = game.check_game_over()

                    #sum_loss += agent1.train_long_term()

                    model_target_update_counter = agent1.trainer.model_target_update_counter
                    avg_count = (count)*0.01 + avg_count*0.99
                    moving_avg = (sum_reward/count)*0.01 + moving_avg*0.99
                    loss_moving_avg = (sum_loss / (sum_count if sum_count else 1)) * 0.01 + loss_moving_avg*0.99
                    vrijeme_treniranja = vrijeme2/(sum_count if sum_count else 1)

            #        if((i+1)%500==0 and i >100):
            #           agent1.save_agent_state(file_path_simple)
                    
                    n_games, epsilon, lr = agent1.get_model_state()

                    print(f"igra: {i}")
                    print(f"vrijeme izvođenja: {vrijeme1/count}")
                    print(f"vrijeme treniranja: {vrijeme_treniranja}")
                    print(f"sum_reward: {moving_avg}")
                    print(f"avg count:{avg_count}")
                    print(f"loss moving_avg: {loss_moving_avg}")
                    print(f"epsilon: {epsilon}")
                    print(f"learning rate: {lr}")
                    print(f"broj_poteza{count}")
                    logger.log({
                        "game": i,
                        "vrijeme_izvodenja": vrijeme1/count,
                        "vrijeme_treniranja": vrijeme_treniranja,
                        "sum_reward": moving_avg,
                        "avg_count": avg_count,
                        "loss_moving_avg": loss_moving_avg,
                        "epsilon": epsilon,
                        "learning_rate": lr,
                        "num_moves": count,
                    })


