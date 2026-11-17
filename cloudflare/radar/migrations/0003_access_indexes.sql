CREATE INDEX IF NOT EXISTS radar_runs_session_created ON radar_runs(session,created_at DESC);
CREATE INDEX IF NOT EXISTS radar_runs_session_phase ON radar_runs(session,phase);
CREATE INDEX IF NOT EXISTS radar_seed_attempts_time ON radar_seed_attempts(attempted_at);
CREATE INDEX IF NOT EXISTS radar_members_active_market ON radar_members(active,market,symbol);
CREATE INDEX IF NOT EXISTS radar_outcomes_evaluation ON radar_outcomes(horizon,exit_model,available_at,signal_id);
