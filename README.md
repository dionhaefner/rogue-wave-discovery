# rogue-wave-discovery
Code for the paper "Machine-Guided Discovery of a Real-World Rogue Wave Model" (Häfner et al., 2023).

## Installation

Requires Python 3.8 or higher.

1. Clone the repository
2. Install the requirements: `pip install -r requirements.txt`

## Download the data

```bash
$ bash get_rogue_data.sh
```

(About 6.2 GB of data will be downloaded.)

## Run the experiments

Execute all 24 experiments and save the results in `experiments/executed`, along with a summary in `results.json`:

```bash
$ cd experiments
$ bash run_all.sh ../data
```

**Note:** This will take a long time (about 2 days on a 16-core machine) and requires at least 64 GB of RAM.

## Reproduce the figures

```bash
$ cd plots
$ bash run_all.sh ../data results.json
```

The figures will be saved in `plots/generated`.

**Note:** This requires at least 32 GB of RAM.