# items/tron.py
from .item import Item

class Tron(Item):
    """
    Predstavlja 'Tron' item koji produžuje zmiju dok je aktivan,
    a zatim je skraćuje kada istekne.
    """
    config = {
        'type': 'tron',
        'affect': 'random',
        'pickUpReward': 50,
        'duration': 15,
        'spawnWeight': 3,
        'symbol': 'T'
    }

    def __init__(self, position, affect):
        """
        Stvara novu instancu Tron itema.
        :param position: Rječnik s 'row' i 'column' koordinatama.
        :param affect: Na koga item utječe ('self', 'enemy', 'both').
        """
        super().__init__(position, Tron.config)
        self.affect = affect
        self.temporary_segments = 0

    def do(self, player):
        """
        Produžuje zmiju za jedan segment i prati koliko segmenata treba ukloniti
        kada efekt istekne.
        :param player: Igrač na kojeg item djeluje.
        """
        # Dodaje segment na kraj tijela
        if player.body:
            player.body.append(player.body[-1])

        self.temporary_segments += 1

        # Kada trajanje dođe do 0 (u `process_players_items` metodi),
        # ukloni nakupljene segmente.
        if self.duration == 0:
            segments_to_remove = max(0, self.temporary_segments)
            player.remove_segments(segments_to_remove)