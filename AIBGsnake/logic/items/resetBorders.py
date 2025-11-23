# items/reset_borders.py
from .item import Item

class ResetBorders(Item):
    config = {
        'type': 'reset-borders',
        'affect': 'map',
        'pickUpReward': 30,
        'duration': 1, # trenutni efekt
        'spawnWeight': 1,
        'symbol': 'B'
    }

    def __init__(self, position, affect):
        super().__init__(position, ResetBorders.config)

    def do(self, player):
        """
        Resetira granice ploče na početne vrijednosti.
        """
        player.game.board.reset_borders()