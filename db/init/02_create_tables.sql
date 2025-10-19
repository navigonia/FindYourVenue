-- -----------------------------------------------------
-- 🧍 USERS-TABELLE
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- -----------------------------------------------------
-- 🔊 NOISE_POINTS-TABELLE (angepasst)
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS noise_points (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    geom geometry(Point, 4326) NOT NULL,
    db_value INTEGER NOT NULL,
    note TEXT,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- -----------------------------------------------------
-- Indexe für Performance
-- -----------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_noise_points_geom ON noise_points USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_noise_points_user_id ON noise_points(user_id);