import pygame
from q_logic import Agent
from simplebot import HeatmapBot
import os
import random
import time
import sys

from visuals import draw_game_state, load_sprites

current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory (AIBG-9.0-master/)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to Python's search path
sys.path.insert(0, parent_dir)
from simple_snake_logic.game import SnakeGame

# === Agents ===

id1 = "id1" 
id2 = "id2"
name1 = "ime1"
name2 = "ime2"
CHECKPOINT_FOLDER = 'model_checkpoints'
file_path_simple = os.path.join(CHECKPOINT_FOLDER, 'save_simple.pth')

#agent1 = Agent(name1)
agent1 = HeatmapBot(name2)
#agent1.load_agent_state(file_path_simple, training=False)


def serialize_game_state(game):
    return {
        'map': game.board.map,
        'players': [
            {
                'name': game.player.name,
                'score': game.player.score,
                'body': game.player.body,
                'activeItems': game.player.active_items,
                'lastMoveDirection': game.player.last_move_direction,
                'nextMoveDirection': game.player.next_move_direction,
            }
            # List comprehension je Pythonov način za ono što je .map() u JavaScriptu

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

moves = ["up","down", "left", "right"]
akcije= [moves[i] for i in[3,3,3,3,3,1,1,1,1,3,3,3,0,0,2,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,3,0,2,2,2,2,2,2]]


for i in range(num_games):
    game = SnakeGame()
    game.add_player({"id": id1, "name": name1})


    GameOver = False

    data = serialize_game_state(game)

    paused = False
    step_once = False

    count = 0

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
            count += 1
            moves = []
            move = {"playerId": id1 , "direction": akcije[count]}
            game.process_moves(move)

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