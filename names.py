order = [
    'OptCMDP',
    'MLE CMDP with Bonus',
    'AbsOptCMDP $\\pi_G$',
    'AlwaysSafe $\\pi_A$',
    'AlwaysSafe $\\pi_T$',
    'AlwaysSafe $\\pi_T$ 0.9ĉ',
    'AlwaysSafe $\\pi_\\alpha$',
    'AlwaysSafe $\\pi_\\alpha$ (fixed flow)',
    'Q-Learning',
    'OptMDP',
]
agents = {
    "OptCMDP"                              : order[0],
    "MLEAgent"                             : order[1],
    "AbsOptCMDP ground"                    : order[2],
    "AbsOptCMDP abs"                       : order[3],
    "SafeAbsOptCMDP (global)"              : order[4],
    "SafeAbsOptCMDP (global) .9ĉ"          : order[5],
    "SafeAbsOptCMDP (adaptive)"            : order[6],
    "SafeAbsOptCMDP (adaptive - fix_flow)" : order[7],
    "Qlearning"                            : order[8],
    "OptMDP"                               : order[9],
}