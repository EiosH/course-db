CREATE TABLE IF NOT EXISTS conversation (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
  session_id VARCHAR(64) NOT NULL,
  turn_index INT UNSIGNED NOT NULL,
  question TEXT NOT NULL,
  answer MEDIUMTEXT NOT NULL,
  summary TEXT NOT NULL,
  summarized_turns INT UNSIGNED NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uk_session_turn (session_id, turn_index),
  KEY idx_session_id (session_id)
);
