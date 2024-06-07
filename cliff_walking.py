import os

from opt_cmdp import OptCMDPAgent
from opt_mdp import OptMDPAgent
from util.training import run_experiments_batch


def main():
    c = 2
    h = 15
    number_of_episodes = 100
    seeds = range(40)
    out_dir = os.path.join('results', os.path.basename(__file__).split('.')[0])
    env_id = "gym_factored:cliff_walking_cost-v0"
    env_kwargs = {"num_rows": 4, "num_cols": 6}

    agents = [
        ('OptCMDP', OptCMDPAgent, {
            'cost_bound': c,
            'horizon': h
        }),
        ('OptMDP', OptMDPAgent, {
            'horizon': h
        }),
    ]
    run_experiments_batch(agents, env_id, env_kwargs, number_of_episodes, out_dir, seeds)


if __name__ == '__main__':
    main()
