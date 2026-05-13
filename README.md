# AuLearn26_Assignments-Group_J
### Members
Jean-Pierre $\mathbb R$unge, $\mathbb C$hristian Jagst, Siraj Bhattarai, Asif Kareem

## Video Presentation of Assignment 1
https://cloud.rz.uni-kiel.de/index.php/s/93ynbz8TtNicWSe

## IMPORTANT: Starting Lines for our `run_ql.py`
We've added some extra options in `run_ql.py`. So in order to run our agents, use these lines.

- **Train the Q-Learner**  
`uv run python run_ql.py --agent ql --mode train --episodes 100 --web --action-mode discrete`

- **Train the SARSA-agent**  
`uv run python run_ql.py --agent sarsa --mode train --episodes 100 --web --action-mode discrete`

- **Additionally save the model (every 10 episodes)**  
`uv run python run_ql.py --agent ql --mode train --episodes 100 --web --save --action-mode discrete`

- **Load our best trained Q-Learner and run in evaluation mode**  
`uv run python run_ql.py --agent ql --mode eval --episodes 20 --web --load models/QLearningAgent/MoveToBeaconDiscreteEnv/260512_2056_QLearningAgent_MoveToBeaconDiscreteEnv.pkl --action-mode discrete`

- **Load our best trained SARSA-agent and run in evaluation mode**
`uv run python run_ql.py --agent sarsa --mode eval --episodes 20 --web --load models/SARSAAgent/MoveToBeaconDiscreteEnv/260512_2139_SARSAAgent_MoveToBeaconDiscreteEnv.pkl --action-mode discrete`


