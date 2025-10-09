import math
import random
from functools import reduce

# Uvoz svih potrebnih klasa za iteme
# Putevi su pretpostavljeni, prilagodite ih strukturi vašeg projekta
from logic.items.goldenApple import GoldenApple
from logic.items.tron import Tron
from logic.items.resetBorders import ResetBorders
from logic.items.shorten import Shorten
from logic.items.katana import Katana
from logic.items.armour import Armour
from logic.items.leap import Leap # Pretpostavka da je leap.js -> speed_up.py
from logic.items.freeze import Freeze
from logic.items.nausea import Nausea
from logic.items.apple import Apple

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from snake_game import SnakeGame

class Spawner:
    """
    Klasa odgovorna za stvaranje elemenata igre (jabuke i itemi) na zrcalnim pozicijama.
    """

    def __init__(self, game):
        """
        Stvara novu instancu Spawner.
        :param game: Referenca na glavnu instancu igre (SnakeGame).
        """
        self.game = game

    def find_valid_spawning_position(self):
        """
        Pronalazi valjane pozicije za stvaranje zrcalnih elemenata, izbjegavajući sudare.
        :return: Rječnik s originalnom i zrcalnom pozicijom, ili None ako pozicija nije pronađena.
        """
        attempts = 0
        max_attempts = 50
        #max_attempts = self.game.num_of_columns * self.game.num_of_rows
        min_distance = 1.5  # Osigurava barem 1 ćeliju dijagonalne udaljenosti

        while attempts < max_attempts:
            attempts += 1
            # random.randrange(n) daje broj od 0 do n-1, ekvivalent Math.floor(Math.random() * n)
            original_row = random.randrange(self.game.num_of_rows)
            original_column = random.randrange(math.floor(self.game.num_of_columns / 2))

            mirrored_row = original_row
            mirrored_column = self.game.num_of_columns - 1 - original_column

            # Provjeri jesu li pozicije unutar granica
            if not self.game.board.is_within_borders({'row': original_row, 'column': original_column}) or \
               not self.game.board.is_within_borders({'row': mirrored_row, 'column': mirrored_column}):
                continue

            # Provjeri jesu li pozicije preblizu glavi bilo kojeg igrača
            is_too_close_to_head = any(
                self._is_pos_too_close_to_player(player, original_row, original_column, mirrored_row, mirrored_column, min_distance)
                for player in self.game.players
            )
            if is_too_close_to_head:
                continue

            # Provjeri sudar s tijelima zmija
            collides_with_snake = any(
                self._collides_with_body(player, original_row, original_column) or
                self._collides_with_body(player, mirrored_row, mirrored_column)
                for player in self.game.players
            )
            if collides_with_snake:
                continue
            
            # Provjeri sudar s itemima
            collides_with_item = any(
                (item.row == original_row and item.column == original_column) or
                (item.row == mirrored_row and item.column == mirrored_column)
                for item in self.game.items
            )
            if collides_with_item:
                continue

            return {'original_row': original_row, 'original_column': original_column,
                    'mirrored_row': mirrored_row, 'mirrored_column': mirrored_column}

        return None
    
    # Pomoćne metode radi čitljivosti
    def _is_pos_too_close_to_player(self, player, r1, c1, r2, c2, min_dist):
        if not player.body: return False
        head = player.body[0]
        # Euklidska udaljenost
        dist_orig = math.sqrt((head['row'] - r1)**2 + (head['column'] - c1)**2)
        dist_mirr = math.sqrt((head['row'] - r2)**2 + (head['column'] - c2)**2)
        return dist_orig <= min_dist or dist_mirr <= min_dist

    def _collides_with_body(self, player, row, col):
        return any(segment['row'] == row and segment['column'] == col for segment in player.body)


    def spawn_mirrored_apples(self):
        """
        Stvara dvije jabuke na zrcalnim pozicijama na ploči.
        """
        position = self.find_valid_spawning_position()
        if not position:
            print("Nije moguće pronaći valjane zrcalne pozicije za stvaranje jabuka.")
            return

        original_apple = Apple({'row': position['original_row'], 'column': position['original_column']}, None)
        mirrored_apple = Apple({'row': position['mirrored_row'], 'column': position['mirrored_column']}, None)

        self.game.items.extend([original_apple, mirrored_apple])

    def spawn_mirrored_items(self):
        """
        Stvara dva itema na zrcalnim pozicijama, odabrana na temelju ponderirane vjerojatnosti.
        """
        position = self.find_valid_spawning_position()
        if not position:
            print("Nije moguće pronaći valjane zrcalne pozicije za stvaranje itema.")
            return

        item_classes = [GoldenApple, Tron, ResetBorders, Shorten, Katana, Armour, Leap, Freeze, Nausea]

        # Izračunaj ukupnu težinu za spawn
        total_spawn_weight = sum(cls.config['spawnWeight'] for cls in item_classes)
        
        # Odaberi klasu itema na temelju težine
        rand_val = random.uniform(0, total_spawn_weight)
        current_spawn_weight = 0
        selected_item_class = None
        for item_class in item_classes:
            current_spawn_weight += item_class.config['spawnWeight']
            if rand_val <= current_spawn_weight:
                selected_item_class = item_class
                break
        
        # Odredi tip utjecaja (affect)
        affect = selected_item_class.config['affect']
        if affect == "random":
            affect_roll = random.random()
            if affect_roll < 0.3:
                affect = "self"
            elif affect_roll < 0.8:
                affect = "enemy"
            else:
                affect = "both"
        
        original_item = selected_item_class({'row': position['original_row'], 'column': position['original_column']}, affect)
        mirrored_item = selected_item_class({'row': position['mirrored_row'], 'column': position['mirrored_column']}, affect)
        
        # Kopiraj svojstva (zbog itema koji imaju slučajna svojstva poput 'Shorten')
        mirrored_item.type = original_item.type
        mirrored_item.symbol = original_item.symbol
        if hasattr(original_item, 'randomDirection'):
             mirrored_item.randomDirection = original_item.randomDirection
        
        self.game.items.extend([original_item, mirrored_item])