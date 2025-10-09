import math
import logic.gameConfig as config

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from snake_game import SnakeGame

class Player:
    """
    Predstavlja igrača u igri zmija (snake).
    """

    def __init__(self, game, player_data):
        """
        Stvara novu instancu igrača.
        :param game: Referenca na glavnu instancu igre (SnakeGame).
        :param player_data: Rječnik s početnim podacima o igraču ('id', 'name').
        """
        self.game = game
        self.id = player_data['id']
        self.name = player_data['name']

        self.score = config.PLAYERS_STARTING_SCORE
        self.body = []
        self.init_body_segments()

        self.active_items = []
        self.last_move_direction = None
        self.next_move_direction = None

    def init_body_segments(self):
        """
        Inicijalizira početnu poziciju i segmente tijela igrača.
        """
        is_first_player = len(self.game.players) == 0

        # Inicijaliziraj poziciju igrača
        start_row_index = math.floor(self.game.num_of_rows / 2)
        start_column_index = (config.PLAYERS_STARTING_LENGTH if is_first_player
                              else self.game.num_of_columns - (config.PLAYERS_STARTING_LENGTH + 1))

        # Dodaj segmente tijela koristeći unshift (insert(0) u Pythonu)
        for i in range(config.PLAYERS_STARTING_LENGTH - 1, -1, -1):
            column = start_column_index - i if is_first_player else start_column_index + i
            # Tijelo je lista rječnika
            self.body.insert(0, {'row': start_row_index, 'column': column})

    def remove_segments(self, num_of_segments):
        """
        Skraćuje tijelo igrača za zadani broj segmenata (minimalna duljina je 1).
        :param num_of_segments: Broj segmenata za uklanjanje s kraja tijela.
        """
        final_length = max(1, len(self.body) - num_of_segments)
        self.body = self.body[:final_length]

    def add_score(self, points):
        """
        Dodaje bodove na rezultat igrača, osiguravajući da ne padne ispod 0.
        :param points: Bodovi za dodavanje (mogu biti negativni).
        """
        self.score = max(0, self.score + points)

    def play_move(self, direction):
        """
        Obrađuje jedan potez za igrača.
        :param direction: Smjer kretanja ('up', 'down', 'left', 'right').
        """
        if self.next_move_direction is not None:
            direction = self.next_move_direction
            self.next_move_direction = None

        if direction == "frozen":
            return  # ignoriraj potez

        self.last_move_direction = direction

        if self.is_reverse_direction(direction):
            self.add_score(-config.REVERSE_DIRECTION_PENALTY)
            return

        if direction not in ["up", "down", "left", "right"]:
            self.add_score(-config.ILLEGAL_MOVE_PENALTY)
            return

        # Napravi plitku kopiju glave za izračun nove pozicije
        new_head_pos = self.body[0].copy()
        if direction == "up":
            new_head_pos['row'] -= 1
        elif direction == "down":
            new_head_pos['row'] += 1
        elif direction == "left":
            new_head_pos['column'] -= 1
        elif direction == "right":
            new_head_pos['column'] += 1

        self.body.insert(0, new_head_pos)

        self.update_score_by_movement_direction()
        self.game.collision_handler.check_for_item_collision(self)
        
        # Ukloni zadnji segment (rep) tek nakon provjere sudara
        self.remove_segments(1)


    def update_score_by_movement_direction(self):
        """
        Ažurira rezultat igrača na temelju kretanja u odnosu na centar ploče.
        """
        if len(self.body) < 2: return

        board_center_row = (self.game.num_of_rows - 1) / 2
        board_center_col = (self.game.num_of_columns - 1) / 2

        new_head_pos = self.body[0]
        old_head_pos = self.body[1] # stara glava, sada vrat

        old_distance = abs(old_head_pos['row'] - board_center_row) + abs(old_head_pos['column'] - board_center_col)
        new_distance = abs(new_head_pos['row'] - board_center_row) + abs(new_head_pos['column'] - board_center_col)

        if new_distance < old_distance:
            self.add_score(config.MOVEMENT_TOWARDS_CENTER_REWARD)
        else:
            self.add_score(config.MOVEMENT_AWAY_FROM_CENTER_REWARD)

    def is_reverse_direction(self, incoming_move_direction):
        """
        Provjerava bi li zadani smjer poteza rezultirao obrnutim kretanjem.
        :param incoming_move_direction: Predloženi smjer kretanja.
        :return: True ako bi potez preokrenuo smjer zmije.
        """
        if len(self.body) < 2: return False

        head = self.body[0]
        neck = self.body[1]

        current_direction = None
        if head['row'] == neck['row']:
            current_direction = "right" if head['column'] > neck['column'] else "left"
        else:
            current_direction = "down" if head['row'] > neck['row'] else "up"

        opposites = {"up": "down", "down": "up", "left": "right", "right": "left"}
        return opposites.get(current_direction) == incoming_move_direction

    def add_or_extend_item(self, item):
        """
        Dodaje ili ažurira efekt itema na igrača.
        :param item: Item koji se dodaje.
        """
        existing_item = next((active_item for active_item in self.active_items if item.type == active_item.type), None)

        if existing_item:
            existing_item.duration = item.duration
        else:
            self.active_items.append(item)