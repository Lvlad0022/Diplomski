import os
import sys
import pygame

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


current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory (AIBG-9.0-master/)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to Python's search path
sys.path.insert(0, parent_dir)

print(current_dir)

SPRITE_FOLDER = os.path.join(current_dir,"sprites" )

SPRITES = {
    "apple": "apple.png",
    "goldenApple": "goldenApple.png",
    "katana": "katana.png",
    "armour": "armour.png",
    "tron": "tron.png",
    "leap": "leap.png",
    "freeze": "freeze.png",
    "shorten": "shorten.png",
    "nausea": "nausea.png",
    "reset": "reset.png",

    "snake-body1": "snake-body1.png",
    "snake-body2": "snake-body2.png",
    "snake-head1": "snake-head1.png",
    "snake-head2": "snake-head2.png"

}

def load_sprites():
    loaded = {}
    for key, filename in SPRITES.items():
        path = os.path.join(SPRITE_FOLDER, filename)
        try:
            image = pygame.image.load(path)
            image = pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE))
            loaded[key] = image
        except Exception as e:
            print(f"Failed to load sprite '{filename}': {e}")
    return loaded

sprites = load_sprites()

def draw_game_state(screen, game_data):
    screen.fill(BLACK)

    board = game_data['map']
    players = game_data['players']

    # Convert head positions to sets for identification
    head_positions = {tuple(p['body'][0].values()): idx for idx, p in enumerate(players)}

    for row_idx, row in enumerate(board):
        for col_idx, cell in enumerate(row):
            if cell is None:
                continue

            rect = pygame.Rect(col_idx * CELL_SIZE, row_idx * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            draw_key = None

            if cell['type'] == 'apple':
                draw_key = 'apple'
            elif cell['type'] == 'goldenApple':
                draw_key = 'goldenApple'
            elif cell['type'] in SPRITES:
                draw_key = cell['type']
            elif cell['type'] == 'snake-body':
                if cell['playerName'] == players[0]['name']:
                    player_id = 1
                else:
                    player_id = 2
                draw_key = f'snake-body{player_id}'

            elif cell['type'] == 'snake-head':
                if cell['playerName'] == players[0]['name']:
                    player_id = 1
                else:
                    player_id = 2
                draw_key = f'snake-head{player_id}'

            if draw_key and draw_key in sprites:
                screen.blit(sprites[draw_key], rect)
            else:
                # fallback if sprite not found
                pygame.draw.rect(screen, WHITE, rect)

    font = pygame.font.SysFont(None, 24)
    score_text = font.render(f"{players[0]['name']} Score: {players[0]['score']}", True, WHITE)
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
