# Game configuration constants

# Maksimalan broj poteza prije forsiranog završetka igre.
GAME_MAX_MOVES = 900

# Broj redaka u mreži igre (MORA BITI NEPARAN BROJ).
BOARD_NUM_OF_ROWS = 25

# Broj stupaca u mreži igre.
BOARD_NUM_OF_COLUMNS = 60

# Početna duljina zmije svakog igrača.
PLAYERS_STARTING_LENGTH = 9

# Početni rezultat za svakog igrača.
PLAYERS_STARTING_SCORE = 1000


# Nagrade i kazne u igri
MOVEMENT_TOWARDS_CENTER_REWARD = 20  # nagrada za kretanje prema centru ploče
MOVEMENT_AWAY_FROM_CENTER_REWARD = -10 # kazna za kretanje od centra
ILLEGAL_MOVE_PENALTY = 50              # kazna za ilegalan potez (smjer), može se koristiti i za timeout
REVERSE_DIRECTION_PENALTY = 30         # kazna za potez koji je suprotan trenutnom smjeru
BODY_SEGMENT_LOSS_PENALTY = 30         # kazna po segmentu izgubljenom zbog smanjivanja granica


# Konfiguracija smanjivanja mape
START_SHRINKING_MAP_AFTER_MOVES = 100  # Broj poteza nakon kojih se mapa počinje smanjivati.
SHRINK_MAP_MOVE_INTERVAL = 10          # Broj poteza između svakog smanjivanja.
MINIMUM_BOARD_SIZE = 20                # Broj stupaca preostalih nakon čega se mapa prestaje smanjivati.


# Stvori item otprilike 1 u X poteza.
MODIFIER_SPAWN_CHANCE = 1 / 10

# U Pythonu nije potreban 'module.exports'.
# Druge datoteke mogu jednostavno uvesti ove konstante s:
# import game_config as config
# i zatim im pristupiti s config.GAME_MAX_MOVES