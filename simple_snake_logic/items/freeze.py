# items/freeze.py
from .item import Item

class Freeze(Item):
    config = {
        'type': 'freeze',
        'affect': 'enemy',
        'pickUpReward': 30,
        'duration': 8,
        'spawnWeight': 4,
        'symbol': 'F'
    }

    def __init__(self, position, affect):
        super().__init__(position, Freeze.config)

    def do(self, player):
        """
        Postavlja sljedeći potez igrača na 'frozen'.
        """
        player.next_move_direction = "frozen"