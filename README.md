# Constrained RL exercise

The file `lp_optimistic.py` contains the code necessary to solve a CMDP using a linear programming approach optimistically.
You must change the implementation to take the uncertainty into account.
You can use the file `test_lp_optimistic.py` to check if your implementation is correct.
After that you should be able to run the experiments with the `cliff_walking.py` script.
Finally, you can analyse the results using the jupyter notebook `cliff_analysis.ipynb`


## usage


1. create a virtualenv and activate it
```bash
cd univr_crl/
python3 -m venv .venv
source .venv/bin/activate
```

2. install the dependencies
```bash
pip install -r requirements.txt
```

3. test the installation
    ```bash
    python test.py 
    ```
    ```
    state [0, 4, 0, 3, 8]   action 2    reward=-1   cost=0
    state [0, 3, 0, 3, 7]   action 3    reward=-1   cost=0
    state [0, 3, 0, 3, 6]   action 6    reward=-10  cost=0
    state [0, 4, 0, 3, 5]   action 2    reward=-1   cost=0
    state [0, 4, 0, 3, 4]   action 1    reward=-1   cost=0
    state [0, 4, 0, 3, 3]   action 5    reward=-10  cost=0
    state [1, 4, 0, 3, 2]   action 0    reward=-1   cost=0
    state [2, 4, 0, 3, 1]   action 0    reward=-1   cost=0
    state [2, 4, 0, 3, 0]   action 5    reward=-20  cost=1
    ```


## files

- planning    
    - `lp.py` provides the linear program implementation to solve a MDP and a CMDP
    - `lp_optimistic.py` provides the linear program implementation to solve a MDP and a CMDP considering the model uncertainty
- constrained RL    
    - `opt_cmdp.py` provides an agent that is able to interact with the environment and use the optimistic linear program to compute a policy for CMDPs
    - `opt_mdp.py` agent for MDPs (simply ignore the cost bound)
- experiments    
    - `cliff_walking.py` run the experiments with OptCMDP and OptMDP in the Cliff Walking environment with costs.
    - `cliff_analysis.ipynb` analyses the results of the Cliff Walking experiments. 
- tests    
    - `test_lp_optimistic.py` useful to test if the implement of `lp_optimistic` is correct
    - `test.py` useful to test the dependencies after installation
