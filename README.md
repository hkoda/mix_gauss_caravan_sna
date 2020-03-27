# mix_gauss_caravan_sna
This is the simulation code used in the manuscirpt,
Hiroki Koda, Ikki Matsuda. "Agent-based simulation for reconstructing social structure by observing collective movements with special reference to single-file movement" *bioRxiv* 2020.03.25.007500; doi: https://doi.org/10.1101/2020.03.25.007500.

## Requirements
This code is worked under python3 or anaconda3 with:
- Python 3.6.9 :: Anaconda, Inc.
- numpy 1.17.4 
- matplotlib 3.1.1
- scipy 1.3.2 
- networkx 2.4
- python-louvain 0.13
- Tested on Mac OS 10.15.3

## Usage
- Main code for the simulation is `collective_movement.py`.
- 2 command arguments are prepared.
- the 1-st argument is a parameter, SD ratio of independenter per depender (see explanations below for details). *Integer*.
- the 2-nd argument is a parameter, experiment number (see explanations below for details). *Integer*.
- Run the following code in your terminal.

```python collective_movement.py arg1 arg2```


## Overviews of the simulations
The aim of our simulations was to examine if the individual-by-individual serial ordered movements, i.e., "animal caravan", provide a sufficient information to reconstruct the cluster organizations, which are typically assumed in the animal social group such as primates. 
We performed the computer simulations by the agent-based models, which mainly contained the four steps: 1) the models generated the animal agents to distribute in the 2-dimensional space with the latent cluster organizations, 2) the generated agents were serially aligned following an assumption of the simple collective movement rule (generating the "caravans"), 3) the social networks were computed by the association index defined by the serial orders of the caravans, and 4) finally we evaluated the cluster organization based on the generated social networks. 

## Schematic illustration of the simulation process

![simulation_process](simulation_process_figure.png)

Schematic illustration of the simulation process. In this example, parameters were set as: n<sub>I</sub> = 5, n<sub>D</sub> = 5, &sigma;<sub>I</sub> = 10, &sigma;<sub>D</sub> = 1, n<sub>exp</sub> = 10. The simulation started from spatial distributions of agents generated from one latent state of social group organization, determined by parameters of the mixed Gaussian processes, n<sub>I</sub>, n<sub>D</sub>, &sigma;<sub>I</sub>, (The two top layers). Then, ”single-file movement” data sets were generated by the process described in the section on ordered alignment of agents. The vertical chains of the 30 circles represents single-file movement (the number of each circle is the agent id). Adjacency matrices <b><i>G<sub>1,2,...10</sub></i></b> were generated from
orders in the single-file movement data set, and were convolved to one adjacency matrix <b><i>G</i></b>, which was passed to the social network analysis. Finally, the social network graph was produced, with clustering of the local community (bottom graph). This flow is the single simulation process, which is run 100 times.
