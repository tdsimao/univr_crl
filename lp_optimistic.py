import os
import numpy as np
import cvxpy as cv

from lp import LinearProgrammingPlanner


class OptimisticLinearProgrammingPlanner(LinearProgrammingPlanner):

    def __init__(self, *args, reward_ci=None, cost_ci=None, transition_ci=None, **kwargs):
        super().__init__(*args, **kwargs)
        if reward_ci is None:
            self.r_ci = np.zeros(shape=self.reward.shape)
        else:
            self.r_ci = reward_ci
            self.r_ci[self.terminal] = 0
        if cost_ci is None:
            self.c_ci = np.zeros(shape=self.cost.shape)
        else:
            self.c_ci = cost_ci
            self.c_ci[self.terminal] = 0
        if transition_ci is None:
            self.t_ci = np.zeros(shape=self.transition.shape)
        else:
            self.t_ci = transition_ci
            self.t_ci[self.terminal] = 0
        self.y = []
        self.reward_ub = np.clip(self.reward + self.r_ci, self.min_reward, self.max_reward)
        self.cost_lb = np.clip(self.cost - self.c_ci, self.min_cost, self.max_cost)
        self.transition_ub = np.clip(self.transition + self.t_ci, 0, 1)
        self.transition_lb = np.clip(self.transition - self.t_ci, 0, 1)

    def instantiate_lp_cvxpy(self):
        # variables
        # y is the occupancy on time step h of the tuple s,a,s'
        self.y = [
            [cv.Variable(shape=(self.na, self.ns), nonneg=True) for s in self.states]
            for h in range(self.horizon)
        ]
        # x is the occupancy on time step h of the tuple s,a
        self.x = [cv.Variable(shape=(self.ns, self.na)) for h in range(self.horizon)]

        # expressions
        self.exp_reward = cv.Constant(0)
        self.exp_cost = cv.Constant(0)

        for h in range(self.horizon):
            self.exp_cost += cv.sum(cv.multiply(self.x[h], self.cost))
            for s in self.states:
                self.exp_reward += cv.sum(cv.multiply(self.x[h][s], self.reward))

        # objective
        obj = cv.Maximize(self.exp_reward)

        # constraints
        if self.horizon > 0:
            constraints = []
            for h in range(self.horizon):
                for s in self.states:
                    constraints.append(self.x[h][s] == cv.sum(self.y[h][s], axis=1))
                    if h > 0:
                        inflow = cv.Constant(0)
                        for t in self.states:
                            inflow += cv.sum(self.y[h - 1][t][:, s])

                        if self.terminal[s]:
                            constraints += [cv.sum(self.y[h][s]) == 0]
                        else:
                            constraints += [inflow == cv.sum(self.y[h][s])]
                    else:
                        constraints += [
                            # outflow == inflow (first time step)
                            cv.sum(self.y[0][s]) == self.isd[s]
                            for s in self.states
                        ]

                    constraints += [
                        self.y[h][s][a] == self.x[h][s, a] * self.transition[s, a]
                        for a in self.actions
                    ]

        else:
            constraints = []
        if self.cost_bound is not None:
            constraints.append(self.exp_cost <= self.cost_bound)

        # problem
        self.lp = cv.Problem(obj, constraints)
