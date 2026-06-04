# Noisy Net Circling Experiment

Environment for testing whether NoisyNet noise magnitude tracks common vs rare
state contexts.

## State

Observation shape is `(2, 20, 20)`:

- 2 channels represent 2 dimensions/contexts.
- Only the active channel contains the map.
- Active channel values:
  - `1`: player
  - `0`: free cell
  - `-1`: wall
- Inactive channels are all zeros.

## Channel Distribution

On reset:

- channel `0`: probability `0.95`
- channel `1`: probability `0.05`

The environment is trained as one continuous stream. A full lap internally
resets the position/channel, but `done` stays `False`.

All channels have the same geometry and reward rules. The only difference is
how often they appear.

## Task

The player moves around a central wall. The map is split into 8 angular sectors.
Reward is given only when the player enters the next expected clockwise sector.
After all 8 sectors are collected, the episode ends.

## Visualization

```powershell
conda run --no-capture-output -n q_learning python -u level_experiments\noisy_net_circling\visualize_env.py
```

Force a rare channel:

```powershell
conda run --no-capture-output -n q_learning python -u level_experiments\noisy_net_circling\visualize_env.py --channel 7
```

Use random actions:

```powershell
conda run --no-capture-output -n q_learning python -u level_experiments\noisy_net_circling\visualize_env.py --random-actions
```

## Training

Default training is 200,000 continuous steps with constant learning rate.

```powershell
conda run --no-capture-output -n q_learning python -u level_experiments\noisy_net_circling\train.py
```

Short smoke run:

```powershell
conda run --no-capture-output -n q_learning python -u level_experiments\noisy_net_circling\train.py --num-steps 1200 --log-every 400 --save-every 0 --no-save
```
