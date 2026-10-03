import numpy as np

CROPS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
PRODUCTS = CROPS + ["EGG", "MILK", "WOOL", "FERTILIZER"]
SHOPS = [
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET"
]

def _farm_vector(farm):
    tiles = farm.get("tiles", [])
    counts = np.zeros(7, dtype=np.float32)

    for row in tiles:
        for tile in row:
            if tile is None:
                counts[0] += 1
            elif tile == "LOCKED":
                counts[1] += 1
            elif isinstance(tile, dict):
                kind = tile.get("kind")
                if kind == "WEED":
                    counts[2] += 1
                elif kind == "PLANT":
                    counts[3] += 1
                elif kind in ("COOP", "PASTURE"):
                    counts[4] += 1
                    if tile.get("animal"):
                        counts[5] += 1

    counts[6] = len(farm.get("hands", []))

    board_area = max(1, len(tiles) * len(tiles))
    return np.concatenate([
        counts / board_area,
        np.asarray([farm.get("money", 0.0) / 10000.0], dtype=np.float32),
        np.asarray(farm.get("farmer", [0, 0]), dtype=np.float32) / 10.0,
    ])

def public_features(obs):
    me = obs["farms"][obs["player"]]
    opp = obs["farms"][1 - obs["player"]]

    prices = obs.get("market", {}).get("prices", {})
    inventory = obs.get("market", {}).get("inventory", {})

    market_prices = np.asarray(
        [prices.get(p, 0) for p in PRODUCTS], dtype=np.float32
    ) / 300.0

    market_inventory = np.asarray(
        [inventory.get(p, 0) for p in PRODUCTS], dtype=np.float32
    ) / 10000.0

    shops = np.zeros(len(SHOPS), dtype=np.float32)
    for shop in obs.get("town", {}).get("unlocked_shops", []):
        if shop in SHOPS:
            shops[SHOPS.index(shop)] += 1.0

    scalars = np.asarray([
        obs.get("step", 0) / 720.0,
        obs.get("day", 0) / 30.0,
        obs.get("hour", 0) / 24.0,
        me.get("hires_today", 0) / 8.0,
    ], dtype=np.float32)

    return np.concatenate([
        scalars,
        _farm_vector(me),
        _farm_vector(opp),
        market_prices,
        market_inventory,
        shops / 8.0,
    ]).astype(np.float32)

def private_features(obs):
    private = obs.get("private", {})
    shed = private.get("shed", {})
    seeds = private.get("seeds", {})

    return np.concatenate([
        np.asarray([shed.get(x, 0) for x in PRODUCTS], dtype=np.float32) / 100.0,
        np.asarray([seeds.get(x, 0) for x in CROPS], dtype=np.float32) / 50.0,
    ]).astype(np.float32)

def opponent_target_from_full_observation(full_obs):
    # Used only when a training dataset contains the hidden opponent state.
    opponent = full_obs["private_opponent"]
    shed = opponent.get("shed", {})
    seeds = opponent.get("seeds", {})

    return np.concatenate([
        np.asarray([shed.get(x, 0) for x in PRODUCTS], dtype=np.float32) / 100.0,
        np.asarray([seeds.get(x, 0) for x in CROPS], dtype=np.float32) / 50.0,
    ]).astype(np.float32)
