# items/nausea.py
import random
from .item import Item

class Nausea(Item):
    config = {
        'type': 'nausea',
        'affect': 'enemy',
        'pickUpReward': 90,
        'duration': 1,
        'spawnWeight': 7,
        'symbol': 'N'
    }

    def __init__(self, position, affect):
        super().__init__(position, Nausea.config)
        self.random_direction = random.choice(['up', 'down', 'left', 'right'])

    def do(self, player):
        """
        Forsira sljedeći potez igrača u nasumičnom smjeru.
        """
        player.next_move_direction = self.random_direction