import numpy as np

# from gym_factored.envs.base import DiscreteEnv
# from opt_cmdp import OptCMDPAgent

from lp_optimistic import OptimisticLinearProgrammingPlanner
# from util.mdp import get_mdp_functions

np.seterr(invalid='ignore', divide='ignore')
from opt_cmdp import OptCMDPAgent



class OptMDPAgent(OptCMDPAgent):
    def __init__(self,
                 ns: int,
                 na: int,
                 terminal: np.array,
                 isd: np.array,
                 env,
                 max_reward, min_reward,
                 max_cost, min_cost,
                 horizon=3,
                 cost_bound=None,
                 solver='grb',  # grb, cvxpy
                 verbose=False):
        self.ns, self.na = ns, na
        self.terminal = terminal
        self.isd = isd
        self.env = env
        self.horizon = horizon
        self.cost_bound = cost_bound
        self.verbose = verbose

        self.max_reward = max(max_reward, 0)
        self.min_reward = min_reward
        self.max_cost = max_cost
        self.min_cost = min(min_cost, 0)
        if terminal.any():
            self.max_reward = max(self.max_reward, 0)
            self.min_cost = min(self.min_cost, 0)

        self.estimated_transition = np.full((ns, na, ns), fill_value=1/ns)
        self.estimated_reward = np.full((ns, na), fill_value=self.max_reward)
        self.estimated_cost = np.full((ns, na), fill_value=self.min_cost)
        self.ensure_terminal_states_are_absorbing()

        self.counter_sas = np.zeros((ns, na, ns), dtype=int)
        self.new_counter_sas = np.zeros((ns, na, ns), dtype=int)
        self.acc_reward = np.zeros((ns, na))
        self.acc_cost = np.zeros((ns, na))

        self.solver = solver

        self.planner = OptimisticLinearProgrammingPlanner(
            self.estimated_transition, self.estimated_reward, self.estimated_cost, self.terminal, self.isd,
            self.env, self.max_reward, self.min_reward, self.max_cost, self.min_cost,
            cost_bound=None, horizon=self.horizon,
            transition_ci=np.full((self.ns, self.na, self.ns), fill_value=1.0),
            reward_ci=np.full((self.ns, self.na), fill_value=self.max_reward-self.min_reward),
            cost_ci=np.full((self.ns, self.na), fill_value=self.max_cost-self.min_cost),
            solver=self.solver,
            verbose=self.verbose
        )

        # computing initial policy
        self.planner.solve()
