# items/armour.py
from .item import Item


class Armour(Item):
    config = {
        'type': 'armour',
        'affect': 'self',
        'pickUpReward': 60,
        'duration': 15,
        'spawnWeight': 7,
        'symbol': 'A'
    }

    def __init__(self, position, affect):
        super().__init__(position, Armour.config)

    def do(self, player):
        """
        Armour nema aktivni 'do' efekt, njegov efekt se provjerava
        prilikom sudara (npr. u Katana klasi).
        """
        pass