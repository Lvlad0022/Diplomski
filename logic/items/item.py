# item.py

class Item:
    """
    Bazna klasa za sve iteme u igri.
    """
    def __init__(self, position, config):
        """
        Stvara novu instancu Itema.
        :param position: Rječnik s 'row' i 'column' koordinatama.
        :param config: Rječnik s konfiguracijom za item.
        """
        self.row = position.get('row')
        self.column = position.get('column')

        self.type = config.get('type', 'unknown')
        self.affect = config.get('affect')
        self.pickUpReward = config.get('pickUpReward')
        self.duration = config.get('duration')
        self.spawnWeight = config.get('spawnWeight')
        self.symbol = config.get('symbol', '?')

    def do(self, player):
        """
        Virtualna metoda koju implementiraju podklase.
        Definira što se događa kada igrač pokupi item.
        :param player: Igrač koji je pokupio item.
        """
        raise NotImplementedError('Metoda "do" mora biti implementirana u podklasama')