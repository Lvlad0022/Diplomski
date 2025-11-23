# Pretpostavljamo da su ove datoteke također prevedene u Python
import logic.gameConfig as config
from logic.items.apple import Apple

# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from snake_game import SnakeGame
#     from player import Player

class CollisionHandler:
    """
    Upravlja svom logikom vezanom za sudare u igri zmija.
    """

    def __init__(self, game):
        """
        Stvara novu instancu CollisionHandler.
        :param game: Referenca na glavnu instancu igre (SnakeGame).
        """
        self.game = game

    def check_for_item_collision(self, player):
        """
        Provjerava sudara li se glava igrača s itemom.
        :param player: Igrač za kojeg se provjerava sudar.
        """
        if not player.body:
            return

        new_head_position = player.body[0]
        item_index = -1
        
        # Python nema findIndex, pa koristimo petlju s enumerate
        for i, item in enumerate(self.game.items):
            if item.row == new_head_position["row"] and item.column == new_head_position["column"]:
                item_index = i
                break

        # ako je igrač udario u item, vrati True
        if item_index != -1:
            item = self.game.items[item_index]
            item.has_collided = True  # pripremi za uklanjanje

            player.add_score(item.pickUpReward)

            if item.affect in ["self", "both", "map"]:
                if item.affect == "both":
                    item.affect = "self"
                player.add_or_extend_item(item)

            if item.affect in ["enemy", "both"]:
                # Pronađi drugog igrača
                other_player = next((p for p in self.game.players if p.id != player.id), None)
                if other_player:
                    if item.affect == "both":
                        item.affect = "self" # Vraćamo na self za slučaj da se objekt itema ponovno koristi
                    other_player.add_or_extend_item(item)

    def check_for_wall_collision(self, player):
        """
        Provjerava je li igrač udario u zid.
        :param player: Igrač za kojeg se provjerava.
        :return: True ako se dogodio fatalan sudar sa zidom, inače False.
        """
        if not player.body:
            return True

        head = player.body[0]

        if not self.game.board.is_within_borders(head):
            print(f"Igrač {player.name} je umro udarcem u zid (glava izvan granica).")
            return True

        # Pronađi indeks prvog segmenta koji je udario u zid
        first_wall_index = -1
        for i, segment in enumerate(player.body):
            if not self.game.board.is_within_borders(segment):
                first_wall_index = i
                break
        
        if first_wall_index == -1:
            return False

        # Pohrani odspojene segmente prije uklanjanja
        disconnected_segments = player.body[first_wall_index:]

        # Ukloni sve segmente od prvog koji je u zidu do repa
        player.body = player.body[:first_wall_index]

        # Pretvori odspojene segmente u jabuke
        new_apples = [
            Apple({'row': segment["row"], 'column': segment["column"]}, None)
            for segment in disconnected_segments
            if self.game.board.is_within_borders(segment)
        ]
        self.game.items.extend(new_apples)

        player.score -= len(disconnected_segments) * config.BODY_SEGMENT_LOSS_PENALTY
        return False

    def check_for_player_collision(self, player):
        """
        Provjerava je li igrač udario sam u sebe ili u drugog igrača.
        :param player: Igrač za kojeg se provjerava.
        :return: True ako se sudar dogodio, inače False.
        """
        if not player.body:
            return False
        
        head = player.body[0]

        # Provjera sudara sa samim sobom (preskačemo glavu i prvi segment tijela)
        # .some() u JS je ekvivalent any() u Pythonu
        body_segment = {}
        body_segment["row"] = head["row"] 
        body_segment["column"] = head["column"]
        player_collided_with_self = (body_segment in player.body[2:0])
        if player_collided_with_self:
            print(f"Igrač {player.name} je umro sudarivši se sa samim sobom.")
            return True

        # Pronađi drugog igrača
        other_player = next((p for p in self.game.players if p.id != player.id), None)
        if not other_player or not other_player.body:
            return False

        # Provjera sudara s drugim igračem
        segment ={}
        segment["row"] = head["row"] 
        segment["column"] = head["column"]
        player_collided_with_other_player = (segment in other_player.body)
        if player_collided_with_other_player:
            print(f"Igrač {player.name} je umro sudarivši se s drugim igračem.")
            return True

        return False