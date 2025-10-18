# items/katana.py
import logic.gameConfig as config
from .item import Item
from .apple import Apple

class Katana(Item):
    config = {
        'type': 'katana',
        'affect': 'self',
        'pickUpReward': 60,
        'duration': 10,
        'spawnWeight': 7,
        'symbol': 'K'
    }

    def __init__(self, position, affect):
        super().__init__(position, Katana.config)

    def do(self, player):
        """
        Siječe rep protivničkog igrača i pretvara ga u jabuke.
        """
        player_head = player.body[0]
        other_player = next((p for p in player.game.players if p is not player), None)

        if not other_player or not other_player.body:
            return False

        collision_index = -1
        for i, segment in enumerate(other_player.body):
            if segment['row'] == player_head['row'] and segment['column'] == player_head['column']:
                collision_index = i
                break

        if collision_index in [-1, 0]: # Nema sudara ili je sudar s glavom
            return False

        has_armour = any(item.type == 'armour' for item in other_player.active_items)
        if has_armour:
            return False

        disconnected_segments = other_player.body[collision_index:]
        other_player.body = other_player.body[:collision_index]

        new_apples = [
            Apple({'row': seg['row'], 'column': seg['column']}, None)
            for seg in disconnected_segments
            if not (seg['row'] == player_head['row'] and seg['column'] == player_head['column'])
        ]
        player.game.items.extend(new_apples)

        other_player.score -= len(disconnected_segments) * config.BODY_SEGMENT_LOSS_PENALTY
        self.duration = 0 # Katana se potroši nakon uspješnog udarca