CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode TEXT NOT NULL,
    step INTEGER NOT NULL,
    public_features TEXT NOT NULL,
    private_features TEXT NOT NULL,
    opponent_prediction TEXT NOT NULL,
    strategy_state TEXT NOT NULL,
    action TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_observations_episode_step
ON observations (episode, step);
