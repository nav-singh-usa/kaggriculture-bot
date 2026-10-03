import json
import sqlite3
from pathlib import Path

class ExperienceStore:
    def __init__(self, path="data/experience.db"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode TEXT,
                step INTEGER,
                public_features TEXT,
                private_features TEXT,
                opponent_prediction TEXT,
                strategy_state TEXT,
                action TEXT
            )
            '''
        )
        self.conn.commit()

    def record(self, episode, step, public, private, opponent, strategy, action):
        self.conn.execute(
            '''
            INSERT INTO observations
            (episode, step, public_features, private_features,
             opponent_prediction, strategy_state, action)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                str(episode),
                int(step),
                json.dumps(public.tolist()),
                json.dumps(private.tolist()),
                json.dumps(opponent.tolist()),
                json.dumps(strategy.tolist()),
                json.dumps(action),
            )
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
