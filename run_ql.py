import argparse
import logging
import sys

from absl import flags
from absl import logging as absl_logging

from agents.qlAgent import QLearningAgent
from agents.sarsaAgent import SARSAAgent
from env.move_to_beacon_discrete_env import MoveToBeaconDiscreteEnv
from runner.runner import Runner

# pysc2 routine, do not touch
FLAGS = flags.FLAGS
FLAGS(sys.argv[:1])


def main():
    parser = argparse.ArgumentParser(
        description="Run Q-learning agent or SARSA Agent on MoveToBeaconDiscreteEnv."
    )
    parser.add_argument("--agent", choices=["ql", "sarsa"], required=True) # Changed options for Q-learning or SARSA
    parser.add_argument("--mode", choices=["train", "eval"], default="eval") # Added option if agent should train or just evaluate
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--step-mul", type=int, default=8)
    parser.add_argument("--save", action="store_true") # Added option to save model during training.
    parser.add_argument("--load", type=str) # Added option to load a given model (from path + pickle file) beforehand.
    parser.add_argument("--action-mode", choices=["discrete", "continuous"], default="discrete")
    parser.add_argument(
        "--loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel))
    absl_logging.set_verbosity(getattr(absl_logging, args.loglevel))

    env = MoveToBeaconDiscreteEnv(
        is_visualize=args.visualize,
        enable_web=args.web,
        step_mul=args.step_mul,
        action_mode=args.action_mode,
    )
    # Either training or evaluate
    training_mode = args.mode == "train"
    # Changed to either Q-learner or SARSA
    agent_cls = QLearningAgent if args.agent == "ql" else SARSAAgent


    # Load a  model if given and recreate the agent from that.
    if args.load:
        # Need to give whole path + filename -> e.g. models/QLearningAgent/MoveToBeaconDiscreteEnv/...pkl
        agent = agent_cls.load_model(args.load, not training_mode)
    # Otherwise create new agent
    else:
        if training_mode:
            agent = agent_cls(
                observation_space=env.observation_space,
                action_shape=env.action_shape,
            )
        else:
            # If in eval mode, we don't need to explore (exploring without learning is meaningless)
            agent = agent_cls(
                observation_space=env.observation_space,
                action_shape=env.action_shape,
                epsilon=0,
                epsilon_min=0,
            )


    runner = Runner(
        agent=agent,
        env=env,
        is_training= training_mode,
        # Only save each 10th episode if save option choosed
        save_model_each_episode_num= 10 if args.save else -1,
    )
    runner.run(args.episodes)


if __name__ == "__main__":
    main()
