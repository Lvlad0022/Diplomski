import pygame
from simplebot import HeatmapBot
import os
import random
import time
import sys

from logic.visuals import draw_game_state, load_sprites
from q_logic_snake import SimpleSnakeAgent

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory (AIBG-9.0-master/)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to Python's search path
sys.path.insert(0, parent_dir)
from logic.game import SnakeGame

# === Agents ===

id1 = "id1" 
id2 = "id2"
name1 = "ime1"
name2 = "ime2"
CHECKPOINT_FOLDER = 'model_checkpoints'
file_path_simple = os.path.join(CHECKPOINT_FOLDER, 'save_simple.pth')

simplebot1 = HeatmapBot(name1)
simplebot2 = HeatmapBot(name2)

agent1 = SimpleSnakeAgent(player1_name = name1, snake_i="dueling")
agent1.load_agent_state(r"C:\Users\lovro\Desktop\snake\model_saves\zsave_simple_snake3_log_n3_dueling_memory1_scheduler1_gamma0.85_2025-11-27_10-46-48.pt", training=False)

def serialize_game_state(game):
    return {
        'map': game.board.map,
        'players': [
            {
                'name': player.name,
                'score': player.score,
                'body': player.body,
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


# incijaliziranje pygame objekata
 
CELL_SIZE = 20
MAP_ROWS = 25
MAP_COLS = 60
WIDTH = CELL_SIZE * MAP_COLS
HEIGHT = CELL_SIZE * MAP_ROWS
FPS = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SNAKE1_COLOR = (0, 255, 0)
SNAKE2_COLOR = (0, 0, 255)
APPLE_COLOR = (255, 0, 0)


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AIBG Python Visualizer")
clock = pygame.time.Clock()
sprites = load_sprites()
font = pygame.font.SysFont(None, 24)

num_games = 100


for i in range(num_games):
    game = SnakeGame()
    game.add_player({"id": id1, "name": name1})
    game.add_player({"id": id2, "name": name2})


    GameOver = False

    data = serialize_game_state(game)

    paused = False
    step_once = False

    while not GameOver:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    if paused:
                        step_once = True

        if not paused or step_once:
            moves = []
            print( agent1.get_action(data))
            moves.append({"playerId": id1 , "direction": agent1.get_action(data)})
            moves.append({"playerId": id2 , "direction": simplebot2.get_action(data)})
            game.process_moves(moves)

            data = serialize_game_state(game)

            draw_game_state(screen, data)
            clock.tick(FPS)

            step_once = False

            if not paused:
                clock.tick(FPS)  # Normal speed
        else:
            draw_game_state(screen, data, sprites)
            clock.tick(15)  # Lower FPS when paused
        
        GameOver = game.check_game_over()