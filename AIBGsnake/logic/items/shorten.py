# items/shorten.py
import random
from .item import Item

class Shorten(Item):
    config = {
        'type': 'shorten',
        'affect': 'random',
        'pickUpReward': 30,
        'duration': 1, # trenutni efekt
        'spawnWeight': 4,
        'symbol': 'S'
    }

    def __init__(self, position, affect):
        super().__init__(position, Shorten.config)
        self.affect = affect
        self._randomize_shortening_length()

    def do(self, player):
        """
        Uklanja određeni broj segmenata s tijela igrača.
        """
        try:
            segments_to_remove = int(self.type.split('-')[1])
            player.remove_segments(segments_to_remove)
        except (IndexError, ValueError):
            print(f"Greška: Neispravan format tipa za Shorten item: {self.type}")

    def _randomize_shortening_length(self):
        """
        Nasumično odabire duljinu skraćivanja.
        """
        possible_lengths = [10, 15, 25, 40]
        length = random.choice(possible_lengths)
        type_name = f"shorten-{length}"
        self.type = type_name
        self.symbol = str(length)