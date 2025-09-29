import heapq
from collections import deque
import numpy as np
import random

class HeatmapBot:
    """
    Heuristički bot koji donosi odluke na temelju "heatmap" pristupa.
    Cilj je doći do najhladnije točke na mapi, izbjegavajući vruće (opasne) zone.
    """
    def __init__(self, player_name):
        self.player_name = player_name
        # Parametri za heatmapu definirani na jednom mjestu radi lakšeg podešavanja
        self.HEAT_PARAMS = {
            'border': {'heat': 1000, 'gamma': 0.01},
            'opponent_body': {'heat': 100, 'gamma': 0.2},
            'nausea': {'heat': -40, 'gamma': 0.5},
            'freeze': {'heat': -40, 'gamma': 0.5},
            # Možete dodati i druge objekte, npr. jabuke s malom negativnom vrućinom
            'apple': {'heat': -10, 'gamma': 0.4}
        }
        self.VISION_WINDOW = 10 # 20x20 prozor oko glave

    def __call__(self, game_state):
        """Omogućuje pozivanje instance kao funkcije: agent(stanje)"""
        return self.get_action(game_state)

    def _get_my_head_and_opponent_name(self, game_state):
        """Pomoćna funkcija za pronalazak glave i imena protivnika."""
        my_head = None
        opponent_name = None
        for player in game_state['players']:
            if player['name'] == self.player_name:
                if player['body']:
                    my_head = player['body'][0]
            else:
                opponent_name = player['name']
        return my_head, opponent_name
    
    def _get_current_bounds(self, move_count, rows, cols):
        """
        Vraća rječnik s trenutnim sigurnim granicama mape na temelju broja poteza.
        Ovo je pretpostavka o logici smanjivanja, možda je treba prilagoditi.
        """
        shrink_count = move_count // 10
        min_r = shrink_count
        max_r = rows - 1 - shrink_count
        min_c = shrink_count
        max_c = cols - 1 - shrink_count
        
        return {'min_row': min_r, 'max_row': max_r, 'min_col': min_c, 'max_col': max_c}

    # =========================================================================
    # === OPTIMIZIRANA VERZIJA FUNKCIJE ===
    # =========================================================================
    def _create_heatmap(self,my_head, game_map, opponent_name, move_count):
        """
        Stvara heatmapu koristeći Multi-source BFS za drastično ubrzanje.
        """
        rows, cols = len(game_map), len(game_map[0])
        heatmap = np.zeros((rows, cols), dtype=np.float32)

        r_start = max(0, my_head['row'] - self.VISION_WINDOW // 2)
        r_end = min(rows, my_head['row'] + self.VISION_WINDOW // 2)
        c_start = max(0, my_head['column'] - self.VISION_WINDOW // 2)
        c_end = min(cols, my_head['column'] + self.VISION_WINDOW // 2)

        # --- Logika za predviđanje zida (ostaje ista) ---
        if move_count > 0 and move_count % 10 in [7, 8, 9]:
            next_shrink_move_count = ((move_count // 10) + 1) * 10
            future_bounds = self._get_current_bounds(next_shrink_move_count, rows, cols)
            danger_heat = self.HEAT_PARAMS['border']['heat']
            
            # Gornji i donji red
            if 0 <= future_bounds['min_row'] < rows:
                heatmap[future_bounds['min_row'], :] += danger_heat
            if 0 <= future_bounds['max_row'] < rows:
                heatmap[future_bounds['max_row'], :] += danger_heat
            # Lijevi i desni stupac
            if 0 <= future_bounds['min_col'] < cols:
                heatmap[:, future_bounds['min_col']] += danger_heat
            if 0 <= future_bounds['max_col'] < cols:
                heatmap[:, future_bounds['max_col']] += danger_heat
        
        # --- NOVA, BRŽA LOGIKA ŠIRENJA TOPLINE ---
        
        # 1. Prikupljanje izvora po tipu
        sources_by_type = {}
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                cell = game_map[r][c]
                if not cell: continue
                
                cell_type_key = None
                cell_type = cell.get('type')
                if cell_type == 'border':
                    cell_type_key = 'border'
                elif cell_type in ['snake-body', 'snake-head'] and cell.get('playerName') == opponent_name:
                    cell_type_key = 'opponent_body'
                elif cell_type in self.HEAT_PARAMS:
                    cell_type_key = cell_type

                if cell_type_key:
                    if cell_type_key not in sources_by_type:
                        sources_by_type[cell_type_key] = []
                    sources_by_type[cell_type_key].append((r, c))

        # 2. Multi-source BFS za svaki tip izvora
        for source_type, positions in sources_by_type.items():
            params = self.HEAT_PARAMS[source_type]
            heat = params['heat']
            gamma = params['gamma']

            # Preskoči ako nema efekta
            if heat == 0: continue

            q = deque()
            distances = np.full((rows, cols), -1, dtype=np.int16)

            for r, c in positions:
                q.append(((r, c), 0))
                distances[r, c] = 0
            
            # Jedan BFS za sve izvore ovog tipa
            while q:
                (r, c), k = q.popleft()
                
                # Proširi na susjede
                if gamma > 0 and gamma < 1 and abs(heat * (gamma ** k)) > 0.1:
                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and distances[nr, nc] == -1:
                            distances[nr, nc] = k + 1
                            q.append(((nr, nc), k + 1))
            
            # 3. Primijeni toplinu vektorski (super brzo s Numpy)
            valid_distances = distances != -1
            heatmap[valid_distances] += heat * (gamma ** distances[valid_distances])

        return heatmap
        
    def _is_safe(self, pos, game_map):
        """Provjerava je li pozicija sigurna (nije zid ili tijelo bilo koje zmije)."""
        rows, cols = len(game_map), len(game_map[0])
        r, c = pos['row'], pos['column']
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        
        cell = game_map[r][c]
        if cell and cell.get('type') in ['border', 'snake-body', 'snake-head']:
            return False
            
        return True

    def _find_coldest_reachable_target(self, my_head, heatmap, game_map):
        """
        Pronađi najhladniju DOSTIŽNU sigurnu točku unutar prozora za gledanje.
        """
        rows, cols = len(game_map), len(game_map[0])
        reachable_cells = []
        q = deque([my_head])
        visited = { (my_head['row'], my_head['column']) }
        r_start = max(0, my_head['row'] - self.VISION_WINDOW // 2)
        r_end = min(rows, my_head['row'] + self.VISION_WINDOW // 2)
        c_start = max(0, my_head['column'] - self.VISION_WINDOW // 2)
        c_end = min(cols, my_head['column'] + self.VISION_WINDOW // 2)
        while q:
            pos = q.popleft()
            reachable_cells.append(pos)
            for move_delta in [(0,1), (0,-1), (1,0), (-1,0)]:
                next_pos = {'row': pos['row'] + move_delta[0], 'column': pos['column'] + move_delta[1]}
                is_in_window = (r_start <= next_pos['row'] < r_end and c_start <= next_pos['column'] < c_end)
                if is_in_window and self._is_safe(next_pos, game_map):
                    if (next_pos['row'], next_pos['column']) not in visited:
                        visited.add((next_pos['row'], next_pos['column']))
                        q.append(next_pos)

        if not reachable_cells:
            return None

        min_heat = float('inf')
        coldest_target = None
        for pos in reachable_cells:
            if pos['row'] == my_head['row'] and pos['column'] == my_head['column']:
                continue

            heat = heatmap[pos['row']][pos['column']]
            if heat < min_heat:
                min_heat = heat
                coldest_target = pos
                
        return coldest_target

    def _manhattan_distance(self, pos1, pos2):
        """Pomoćna funkcija za izračun Manhattan udaljenosti (heuristika za A*)."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _find_best_path(self, start_pos, end_pos, heatmap, game_map):
        """
        Pronađi "najhladniji" put od start do end koristeći A* algoritam.
        """
        start_node = (start_pos['row'], start_pos['column'])
        end_node = (end_pos['row'], end_pos['column'])
        
        g_score = {start_node: 0}
        f_score = {start_node: self._manhattan_distance(start_node, end_node)}
        
        pq = [(f_score[start_node], [start_node])] # (f_score, path)

        while pq:
            _, path = heapq.heappop(pq)
            current_node = path[-1]

            if current_node == end_node:
                return path 

            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                next_node = (current_node[0] + dr, current_node[1] + dc)
                next_pos_dict = {'row': next_node[0], 'column': next_node[1]}
                
                if self._is_safe(next_pos_dict, game_map):
                    tentative_g_score = g_score[current_node] + heatmap[next_node[0]][next_node[1]]
                    
                    if tentative_g_score < g_score.get(next_node, float('inf')):
                        g_score[next_node] = tentative_g_score
                        h_score = self._manhattan_distance(next_node, end_node)
                        f_score[next_node] = tentative_g_score + h_score
                        
                        new_path = path + [next_node]
                        heapq.heappush(pq, (f_score[next_node], new_path))
        
        return None 

    def get_action(self, game_state):
        """
        Glavna metoda za odlučivanje poteza.
        """
        my_head, opponent_name = self._get_my_head_and_opponent_name(game_state)
        if not my_head: return 'up'

        game_map = game_state['map']
        move_count = game_state['moveCount']
        
        # KORAK 1: Odredi sve sigurne poteze oko glave.
        safe_adjacent_moves = {}
        move_deltas = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        for move, delta in move_deltas.items():
            pos = {'row': my_head['row'] + delta[0], 'column': my_head['column'] + delta[1]}
            if self._is_safe(pos, game_map):
                safe_adjacent_moves[move] = pos
        
        if not safe_adjacent_moves:
            return 'up' 
        
        # KORAK 2: Glavna logika s heatmapom
        heatmap = self._create_heatmap(my_head, game_map, opponent_name, move_count)
        target_pos = self._find_coldest_reachable_target(my_head, heatmap, game_map)
        
        path = None
        if target_pos:
            min_heat = np.min(heatmap)
            pathfinding_heatmap = heatmap - min_heat + 0.01 
            path = self._find_best_path(my_head, target_pos, pathfinding_heatmap, game_map)
        
        # KORAK 3: Odluka o potezu
        if path and len(path) > 1:
            next_step = path[1]
            dr, dc = next_step[0] - my_head['row'], next_step[1] - my_head['column']
            if dr == -1: return 'up'
            if dr == 1: return 'down'
            if dc == -1: return 'left'
            if dc == 1: return 'right'
        
        # --- FALLBACK LOGIKA ---
        best_fallback_move = random.choice(list(safe_adjacent_moves.keys()))
        max_space = -1

        for move, pos in safe_adjacent_moves.items():
            space = self._calculate_free_space(pos, game_map)
            if space > max_space:
                max_space = space
                best_fallback_move = move
        
        return best_fallback_move

    def _calculate_free_space(self, start_pos, game_map):
        if not self._is_safe(start_pos, game_map): return 0
        count = 0
        queue = deque([start_pos])
        visited = { (start_pos['row'], start_pos['column']) }
        while queue:
            pos = queue.popleft()
            count += 1
            for move_delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_pos = {'row': pos['row'] + move_delta[0], 'column': pos['column'] + move_delta[1]}
                if (next_pos['row'], next_pos['column']) not in visited and self._is_safe(next_pos, game_map):
                    visited.add((next_pos['row'], next_pos['column']))
                    queue.append(next_pos)
        return count