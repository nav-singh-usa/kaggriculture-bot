import uuid
import numpy as np

from agent.features import public_features, private_features
from data.store import ExperienceStore
from models.opponent_model import OpponentModel
from models.strategy_model import StrategyModel
from models.move_model import MoveModel

class KaggricultureAgent:
    def __init__(self):
        self.episode = str(uuid.uuid4())
        self.store = ExperienceStore()

        blank = self._blank_obs()
        public_dim = public_features(blank).shape[0]
        private_dim = private_features(blank).shape[0]

        self.opponent = OpponentModel(input_dim=public_dim)
        self.strategy = StrategyModel(
            input_dim=public_dim + 13 + private_dim
        )
        self.move = MoveModel(
            input_dim=32 + 13 + public_dim
        )

    @staticmethod
    def _blank_obs():
        farm = {
            "money": 3000,
            "tiles": [[None] * 10 for _ in range(10)],
            "farmer": [4, 4],
            "hands": [],
            "hires_today": 0,
        }

        return {
            "player": 0,
            "step": 0,
            "day": 0,
            "hour": 0,
            "farms": [farm, farm.copy()],
            "market": {"prices": {}, "inventory": {}},
            "town": {"unlocked_shops": []},
            "private": {
                "shed": {},
                "seeds": {},
                "inventories": [[]]
            }
        }

    def _reset_if_new_episode(self, obs):
        if obs.get("step", 0) == 0:
            self.episode = str(uuid.uuid4())
            self.strategy.reset()

    def __call__(self, obs):
        self._reset_if_new_episode(obs)

        public = public_features(obs)
        private = private_features(obs)

        belief = self.opponent.predict(public)

        strategy_input = np.concatenate([
            public,
            belief,
            private
        ])
        strategy_state = self.strategy.update(strategy_input)

        move_input = np.concatenate([
            strategy_state,
            belief,
            public
        ])
        move = self.move.action(move_input)

        action = self._action_from_move(move)

        self.store.record(
            self.episode,
            obs.get("step", 0),
            public,
            private,
            belief,
            strategy_state,
            action
        )

        return action

    @staticmethod
    def _action_from_move(move):
        if move.startswith("PLANT_"):
            return {
                "farmer": ["PLANT", move.removeprefix("PLANT_")],
                "hands": [],
                "market": []
            }

        return {
            "farmer": [move],
            "hands": [],
            "market": []
        }
