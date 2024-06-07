import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd



order = [
    'OptCMDP',
    'MLE CMDP with Bonus',
    'AbsOptCMDP-$\\pi_G$',
    'AlwaysSafe-$\\pi_\\alpha$',
    'AlwaysSafe-$\\pi_A$',
    'AlwaysSafe-$\\pi_T$',
    'AlwaysSafe-$\\pi_T$ 0.9ĉ',
    'Q-Learning',
    'Q-Learning Optimistic',
    'OptMDP',
    # 'AlwaysSafe $\\pi_\\alpha$ (fixed flow)',
]
agents = {
    "OptCMDP"                              : order[0],
    "MLEAgent"                             : order[1],
    "AbsOptCMDP ground"                    : order[2],
    "SafeAbsOptCMDP (adaptive)"            : order[3],
    "AbsOptCMDP abs"                       : order[4],
    "SafeAbsOptCMDP (global)"              : order[5],
    "SafeAbsOptCMDP (global) .9ĉ"          : order[6],
    "Qlearning"                            : order[7],
    "QLearning"                            : order[7],
    "Optimistic QLearning"                 : order[8],
    # "SafeAbsOptCMDP (adaptive - fix_flow)" : order[7],
    "OptMDP"                              : order[9],
}


def plot_and_save(column, data, out_dir, max_x, ax=None, cost_bound=None, optimal_return=None, hue="Agent", errorbar=('ci', 95), title=None, min_y=None, max_y=None, ncol=2):
#     plt.style.use('./jaamas.mplstyle')
    data = data[data["Episode"] < max_x]
    if ax is None:
        fig = plt.figure()
    ax = sns.lineplot(x="Episode", y=column, hue=hue,
                      data=data, hue_order=order,
                      errorbar=errorbar,
                      legend='Cost' in column,
                      ax=ax
                      )
    if 'Cost' in column:
        handles, labels = ax.get_legend_handles_labels()
        indexes = [labels.index(l) for l in labels if l in set(data[hue].unique())]
        leg = ax.legend(
            handles=[handles[i] for i in indexes],
            labels=[labels[i] for i in indexes],
            ncol=ncol,
       )
        # for legobj in leg.legendHandles:
        #     legobj.set_linewidth(3.0)
        ax.add_artist(leg)
        l1, = ax.plot(np.arange(max_x),
                 np.ones(max_x)*cost_bound,
                 linestyle='dashed', marker='', color='black',
                 label='cost bound')
    else:
        l1, = ax.plot(np.arange(max_x),
                 np.ones(max_x)*optimal_return,
                 linestyle=':', marker='', color='black',
                 label='optimal return')
        
    ax.legend(handles=[l1], ncol=1,
              loc='lower right',
              borderpad=0.0,
              borderaxespad=0.2,
              )
    ax.set_title(title)
    # for legobj in leg.legendHandles:
    #     legobj.set_linewidth(3.0)
    
    ax.set_xlim(0, max_x)
    if min_y is not None:
        ax.set_ylim(min_y, max_y)
#     if ax is None:
#         file_name = column.replace(" ", "_").lower()
#         fig.savefig(os.path.join(out_dir, file_name + '.pdf'), dpi=300, bbox_inches='tight')


def load_results(number_of_episodes, results_dir, window_size, agents=agents):
    dfs = []
    s = "{:>38} {}"
    print(s.format("agent", "seeds"))
    for agent_name in os.listdir(results_dir):
        agent_dir = os.path.join(results_dir, agent_name)
        seeds = [results_file.split('_')[-1].split('.')[0] for results_file in glob.glob('{}/*.csv'.format(agent_dir))]
        print(s.format(agent_name, len(seeds)))
    dfs = []
    for agent_name in os.listdir(results_dir):
        if agent_name not in agents:
            continue
        agent_dir = os.path.join(results_dir, agent_name)
        for results_file in glob.glob('{}/*.csv'.format(agent_dir)):
            df = pd.read_csv(results_file, index_col=0)
            seed = results_file.split('_')[-1].split('.')[0]
            df["seed"] = int(seed)
            df["Expected Value"] = df["evaluation_returns"].rolling(window=window_size).mean()
            df["Expected Cost"] = df["evaluation_costs"].rolling(window=window_size).mean()
            df["Agent"] = agents[agent_name]
            df["Episode"] = np.arange(number_of_episodes)
            dfs.append(df)
    df = pd.concat(dfs)
    return df
