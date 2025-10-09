import random
# Pretpostavljamo da su ove datoteke/klase također prevedene u Python
# npr. gameConfig.js -> game_config.py
import logic.gameConfig as config
from logic.player import Player
from logic.board import Board
from logic.spawner import Spawner
from logic.collisionHandler import CollisionHandler

class SnakeGame:
    """
    Glavna klasa igre koja upravlja logikom igre zmija (snake).
    """

    def __init__(self):
        """
        Stvara novu instancu SnakeGame i inicijalizira komponente igre.
        """
        self.num_of_rows = config.BOARD_NUM_OF_ROWS
        self.num_of_columns = config.BOARD_NUM_OF_COLUMNS

        

        self.move_count = 0

        self.players = []
        self.winner = None

        self.items = []

        self.board = Board(self)
        self.spawner = Spawner(self)
        self.collision_handler = CollisionHandler(self)

        self.board.update_map()

    def add_player(self, player_data):
        """
        Dodaje novog igrača u igru.
        :param player_data: Rječnik (dict) s podacima o novom igraču.
                            Očekuje ključeve 'id' i 'name'.
        """
        player = Player(self, player_data)
        self.players.append(player)
        self.board.update_map()

    def process_moves(self, moves):
        """
        Obrađuje niz poteza svih igrača.
        :param moves: Lista (list) rječnika s potezima.
                      Svaki rječnik ima 'player_id' i 'direction'.
        """
        self.move_count += 1

        # Obradi sve poteze
        for move in moves:
            # Pronađi igrača koji odgovara potezu
            player = next((p for p in self.players if p.id == move['playerId']), None)
            if player:
                player.play_move(move['direction'])

        # Obradi iteme koje igrači imaju
        self.process_players_items()

        # Filtriraj sve iteme koji su pokupljeni (imaju has_collided == True)
        self.items = [item for item in self.items if not getattr(item, 'has_collided', False)]

        # Smanjivanje mape
        current_board_width = self.board.get_current_board_width()
        if (current_board_width > config.MINIMUM_BOARD_SIZE and
                self.move_count >= config.START_SHRINKING_MAP_AFTER_MOVES and
                self.move_count % config.SHRINK_MAP_MOVE_INTERVAL == 0):
            self.board.shrink_map()

        # Provjeri je li igra gotova i odredi pobjednika
        self.check_game_over()
        if self.winner is not None:
            return

        # Stvori jabuke svakih 5 poteza
        if self.move_count % 5 == 0:
            self.spawner.spawn_mirrored_apples()

        # Stvori iteme na temelju šanse
        if random.random() < config.MODIFIER_SPAWN_CHANCE:
            self.spawner.spawn_mirrored_items()
        
        # Ažuriraj stanje mape
        self.board.update_map()

    def check_game_over(self):
        """
        Provjerava je li igra gotova na temelju smrti igrača ili ograničenja poteza.
        """
        dead_players_ids = [
            player.id for player in self.players if
            self.collision_handler.check_for_wall_collision(player) or
            self.collision_handler.check_for_player_collision(player) or
            player.score <= 0
        ]

        if len(dead_players_ids) > 0:
            if len(dead_players_ids) == 1:
                # Ako je jedan mrtav, drugi je pobjednik
                winner_player = next(p for p in self.players if p.id not in dead_players_ids)
                self.winner = winner_player.name
                print(f"Kraj igre! Igrač {self.winner} pobjeđuje!")
            else:
                # Ako su oba mrtva, odluči po bodovima/duljini
                self.determine_winner_by_score_then_length()
            return True
        # Provjeri ograničenje poteza samo ako nitko nije umro u ovom potezu
        elif self.move_count >= config.GAME_MAX_MOVES:
            print("Dosegnut je maksimalan broj poteza.")
            self.determine_winner_by_score_then_length()
            return True
        return False
        

    def determine_winner_by_score_then_length(self):
        """
        Određuje pobjednika na temelju bodova i duljine zmije kada je neriješeno.
        Postavlja self.winner na ime pobjednika ili -1 za neriješeno.
        """
        player1, player2 = self.players

        if player1.score != player2.score:
            self.winner = player1.name if player1.score > player2.score else player2.name
            print(f"Kraj igre! Igrač {self.winner} pobjeđuje s više bodova!")
        elif len(player1.body) != len(player2.body):
            self.winner = player1.name if len(player1.body) > len(player2.body) else player2.name
            print(f"Kraj igre! Igrač {self.winner} pobjeđuje s dužom zmijom!")
        else:
            self.winner = -1
            print("Kraj igre! Neriješeno! Jednaki bodovi i duljina.")
    
    def process_players_items(self):
        """
        Obrađuje sve aktivne iteme, smanjuje im trajanje i upravlja efektima.
        """
        for player in self.players:
            for active_item in player.active_items:
                active_item.duration -= 1
                active_item.do(player)
            
            # Uklanja iteme kojima je isteklo trajanje
            player.active_items = [item for item in player.active_items if item.duration > 0]