import math
# Pretpostavljamo da je gameConfig.js preveden u game_config.py
import logic.gameConfig as config
import numpy as np

# U Pythonu nije potrebno uvoziti klase samo radi 'type-hintinga' u docstringovima
# ali da je potrebno, izgledalo bi ovako:
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from snake_game import SnakeGame

class Board:
    """
    Predstavlja ploču za igru zmija (snake).
    """

    def __init__(self, game):
        """
        Stvara novu instancu ploče (Board).
        :param game: Referenca na glavnu instancu igre (SnakeGame).
        """
        self.game = game
        self.horizontal_shrink_level = -1

        self.borders = {
            'left': self.horizontal_shrink_level,
            'right': self.game.num_of_columns - self.horizontal_shrink_level - 1,
            'top': self.horizontal_shrink_level,
            'bottom': self.game.num_of_rows - self.horizontal_shrink_level - 1,
        }

        self.map = None
        self.update_map()

    def get_cell(self, row, column):
        """
        Dohvaća vrijednost ćelije na zadanim koordinatama.
        :param row: Koordinata retka.
        :param column: Koordinata stupca.
        :return: Vrijednost ćelije (objekt) ili None za prazne ćelije.
        """
        if 0 <= row < self.game.num_of_rows and 0 <= column < self.game.num_of_columns:
            return self.map[row][column]
        return None

    def set_cell(self, row, column, value):
        """
        Postavlja vrijednost ćelije na zadanim koordinatama.
        :param row: Koordinata retka.
        :param column: Koordinata stupca.
        :param value: Vrijednost koja se postavlja u ćeliju.
        """
        if 0 <= row < self.game.num_of_rows and 0 <= column < self.game.num_of_columns:
            self.map[row][column] = value

    def get_current_board_width(self):
        """
        Dohvaća trenutnu širinu igrive površine ploče.
        :return: Trenutna širina ploče bez zidova.
        """
        return self.borders['right'] - self.borders['left'] - 1

    def shrink_map(self):
        """
        Smanjuje ploču za igru povećanjem razine smanjivanja i ažuriranjem granica.
        """
        self.horizontal_shrink_level += 1

        # Ažuriraj granice
        self.borders['left'] = self.horizontal_shrink_level
        self.borders['right'] = self.game.num_of_columns - 1 - self.horizontal_shrink_level

        # Izračunaj koliko je dodatnog vertikalnog smanjivanja potrebno
        vertical_shrink = max(
            self.horizontal_shrink_level - math.floor((self.game.num_of_columns - self.game.num_of_rows) / 2),
            -1
        )
        self.borders['top'] = vertical_shrink
        self.borders['bottom'] = self.game.num_of_rows - 1 - vertical_shrink

        # Ukloni iteme izvan novih granica
        self.game.items = [item for item in self.game.items if self.is_within_borders(item)]

    def is_within_borders(self, position):
        """
        Provjerava je li pozicija unutar trenutnih granica ploče.
        :param position: Objekt s 'row' i 'column' svojstvima.
        :return: True ako je pozicija unutar granica, inače False.
        """
        # U Pythonu, pozicija bi mogla biti rječnik ili objekt. Pristupamo kao rječniku.
        # Ako je objekt, pristup bi bio position.row, position.column
        pos_row = position['row'] if isinstance(position, dict) else position.row
        pos_col = position['column'] if isinstance(position, dict) else position.column
        
        return (self.borders['left'] < pos_col < self.borders['right'] and
                self.borders['top'] < pos_row < self.borders['bottom'])

    def update_map(self):
        """
        Ažurira cijelu mapu igre s trenutnim stanjem igre.
        """
        # Inicijaliziraj rešetku (grid)
        self.map = [
            [
                {'type': 'border', 'symbol': '#'} if
                col_idx <= self.borders['left'] or
                col_idx >= self.borders['right'] or
                row_idx <= self.borders['top'] or
                row_idx >= self.borders['bottom']
                else None
                for col_idx in range(self.game.num_of_columns)
            ]
            for row_idx in range(self.game.num_of_rows)
        ]

        # Ažuriraj iteme
        if self.game.items:
            for item in self.game.items:
                # Ovdje pretpostavljamo da je 'item' objekt sa svojstvima
                self.set_cell(item.row, item.column, {
                    'row': item.row,
                    'column': item.column,
                    'type': getattr(item, 'type', 'unknown'),
                    'affect': getattr(item, 'affect', None),
                    'pickUpReward': getattr(item, 'pickUpReward', 0),
                    'duration': getattr(item, 'duration', 0),
                    'spawnWeight': getattr(item, 'spawnWeight', 0),
                    'symbol': getattr(item, 'symbol', '?'),
                })

        # Ažuriraj igrače
        if self.game.players:
            for player in self.game.players:
                # Crtaj segmente zmije od repa prema glavi
                for i in range(len(player.body) - 1, -1, -1):
                    snake_segment = player.body[i]
                    if not self.is_valid_position(snake_segment):
                        break
                    
                    pos_row = snake_segment['row'] if isinstance(snake_segment, dict) else snake_segment.row
                    pos_col = snake_segment['column'] if isinstance(snake_segment, dict) else snake_segment.column

                    self.set_cell(pos_row, pos_col, {
                        'type': 'snake-head' if i == 0 else 'snake-body',
                        'playerName': player.name,
                    })

    def is_valid_position(self, position):
        """
        Provjerava je li pozicija valjana unutar dimenzija ploče.
        :param position: Objekt ili rječnik s 'row' i 'column' svojstvima.
        :return: True ako je pozicija valjana, inače False.
        """
        if not position:
            return False
            
        pos_row = position['row'] if isinstance(position, dict) else position.row
        pos_col = position['column'] if isinstance(position, dict) else position.column

        return (0 <= pos_row < self.game.num_of_rows and
                0 <= pos_col < self.game.num_of_columns)

    def reset_borders(self):
        """
        Resetira razinu smanjivanja i granice na početne vrijednosti.
        """
        self.horizontal_shrink_level = -1
        self.borders = {
            'left': self.horizontal_shrink_level,
            'right': self.game.num_of_columns - self.horizontal_shrink_level - 1,
            'top': self.horizontal_shrink_level,
            'bottom': self.game.num_of_rows - self.horizontal_shrink_level - 1,
        }