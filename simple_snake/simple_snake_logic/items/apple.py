# items/apple.py
from .item import Item


class Apple(Item):
    config = {
        'type': 'apple',
        'affect': 'self',
        'pickUpReward': 50,
        'duration': 1,
        'symbol': 'A'
    }

    def __init__(self, position, affect):
        super().__init__(position, Apple.config)

    def do(self, player):
        """
        Implementira efekt skupljanja jabuke - produžuje zmiju za 1.
        """
        # Dodaje segment na kraj tijela koji se neće ukloniti u istom potezu
        if player.body:
            player.body.append(player.body[-1])