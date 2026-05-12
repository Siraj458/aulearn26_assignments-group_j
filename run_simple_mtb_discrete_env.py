import argparse
import logging
import sys

from absl import flags
from absl import logging as absl_logging

from agents.basicAgent import BasicAgent
from agents.randomAgent import RandomAgent
from env.move_to_beacon_discrete_env import MoveToBeaconDiscreteEnv
from runner.runner import Runner

# pysc2 routine, do not touch
FLAGS = flags.FLAGS
FLAGS(sys.argv[:1])


def main():
    parser = argparse.ArgumentParser(
        description="Run BasicAgent or RandomAgent on MoveToBeaconDiscreteEnv."
    )
    parser.add_argument("--agent", choices=["basic", "random"], required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--step-mul", type=int, default=8)
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

    agent_cls = BasicAgent if args.agent == "basic" else RandomAgent
    agent = agent_cls(
        observation_space=env.observation_space,
        action_shape=env.action_shape,
    )

    runner = Runner(
        agent=agent,
        env=env,
        is_training=False,
        save_model_each_episode_num=-1,
    )
    runner.run(args.episodes)


if __name__ == "__main__":
    main()
