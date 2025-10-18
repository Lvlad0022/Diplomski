# items/leap.py
from .item import Item

class Leap(Item):
    config = {
        'type': 'leap',
        'affect': 'random',
        'pickUpReward': 80,
        'duration': 5,
        'spawnWeight': 5,
        'symbol': 'J'
    }

    def __init__(self, position, affect):
        super().__init__(position, Leap.config)

    def do(self, player):
        """
        Uzrokuje da igrač napravi dvostruki potez u jednom koraku.
        Također uklanja 'katana' item ako ga igrač ima.
        """
        player.play_move(player.last_move_direction)
        player.active_items = [item for item in player.active_items if item.type != 'katana']