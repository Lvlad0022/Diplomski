# items/golden_apple.py
from .item import Item

class GoldenApple(Item):
    config = {
        'type': 'golden-apple',
        'affect': 'self',
        'pickUpReward': 70,
        'duration': 5,
        'spawnWeight': 7,
        'symbol': 'G'
    }

    def __init__(self, position, affect):
        super().__init__(position, GoldenApple.config)

    def do(self, player):
        """
        Implementira efekt skupljanja zlatne jabuke.
        """
        if player.body:
            player.body.append(player.body[-1])