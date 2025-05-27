# db_manager.py

import os, sqlite3, json, time, shutil
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.getcwd())  # Project root directory (absolute path)

class DBManager:
    """
    Unified SQLite Manager:
      - register_model: Optional dir_path. If omitted, automatically creates ./out/{name}_{id}/
      - rename_model: Synchronously rename both ./data and ./out directories
      - delete_model: Synchronously delete both ./data and ./out directories
      - get_model_basic_info: Returns {'name', 'dir_path'}
    """
    # ------------------------------------------------------------------ #
    # Initialization & Table Creation
    # ------------------------------------------------------------------ #
    def __init__(self, db_path: str = "assets/model_registry.db"):
        """
        Initialize the database connection and ensure all tables exist.
        """
        rel_db_path = os.path.relpath(db_path, PROJECT_ROOT)
        self.conn = sqlite3.connect(rel_db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """
        Create tables for models, training configs, training logs,
        inference configs, and inference history if they do not exist.
        """
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS models(
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                dir_path TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS training_configs(
                model_id INTEGER PRIMARY KEY,
                config_json TEXT NOT NULL,
                FOREIGN KEY(model_id) REFERENCES models(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS training_logs(
                model_id INTEGER PRIMARY KEY,
                log_path TEXT NOT NULL,
                FOREIGN KEY(model_id) REFERENCES models(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS inference_configs(
                model_id INTEGER PRIMARY KEY,
                config_json TEXT NOT NULL,
                FOREIGN KEY(model_id) REFERENCES models(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS inference_history(
                model_id INTEGER PRIMARY KEY,
                content TEXT,
                FOREIGN KEY(model_id) REFERENCES models(id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Internal Path Utilities
    # ------------------------------------------------------------------ #
    def _rel(self, path: str) -> str:
        """
        Convert an absolute path to a path relative to the project root.
        """
        return os.path.relpath(path, PROJECT_ROOT)

    def _abs(self, rel_path: str) -> str:
        """
        Convert a relative path (from project root) to an absolute path.
        """
        return os.path.join(PROJECT_ROOT, rel_path) if rel_path else ""

    # ------------------------------------------------------------------ #
    # Model Registration & Basic Info
    # ------------------------------------------------------------------ #
    def register_model(self, name: str, dir_path: Optional[str] = None) -> int:
        """
        Register a new model or return its ID if it already exists.

        Key Fixes:
        1. Unified Timestamp:
           Previous versions called `time.time()` separately for
           directory name and ID, causing inconsistency between
           directory names and IDs. Now a single timestamp is
           used for both:
             - Directory suffix (e.g., out/foobar_1747405262784/)
             - Database primary key ID
           Ensures consistency across directory structure, IDs, and UI.

        2. Existing Directory Handling:
           If the provided `dir_path` is already registered,
           return its ID directly and do not attempt to update
           its name, avoiding unintended renaming to long
           strings like “test_1747405262784”.

        Parameters:
        name: Display name for the model (visible in UI)
        dir_path: Optional. If not provided, automatically generates
                  ./out/{name}_{id}/ and registers it.

        Returns:
        int: Unique model ID (also used as the directory suffix)
        """
        cur = self.conn.cursor()

        # ----- Case 1: Provided directory, check if already registered -----
        if dir_path:
            rel_path = self._rel(dir_path)
            cur.execute("SELECT id FROM models WHERE dir_path = ?", (rel_path,))
            row = cur.fetchone()
            if row:
                # Already registered: return existing ID
                return row["id"]

            # Not registered: ensure directory exists then proceed
            abs_path = self._abs(rel_path)
            if not os.path.exists(abs_path):
                os.makedirs(abs_path, exist_ok=True)

            # Unified timestamp: used for both directory suffix and ID
            ts_ms = int(time.time() * 1000)
            cur.execute(
                "INSERT INTO models(id, name, dir_path, created_at) "
                "VALUES(?,?,?,datetime('now'))",
                (ts_ms, name, rel_path)
            )
            self.conn.commit()
            return ts_ms

        # ----- Case 2: No directory provided, auto-generate -----
        ts_ms = int(time.time() * 1000)
        auto_folder = f"{name}_{ts_ms}"
        dir_path_rel = self._rel(os.path.join("out", auto_folder))
        os.makedirs(self._abs(dir_path_rel), exist_ok=True)

        cur.execute(
            "INSERT INTO models(id, name, dir_path, created_at) "
            "VALUES(?,?,?,datetime('now'))",
            (ts_ms, name, dir_path_rel)
        )
        self.conn.commit()
        return ts_ms
    
    def get_model_id_by_dir(self, dir_path: str) -> Optional[int]:
        """
        Retrieve the model ID based on its directory path.
        """
        dir_path_rel = self._rel(dir_path)
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM models WHERE dir_path = ?", (dir_path_rel,))
        row = cur.fetchone()
        return row["id"] if row else None
    
    def get_model_basic_info(self, model_id: int) -> Optional[dict]:
        """
        Return basic information for a model, including its name and directory path.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT name, dir_path FROM models WHERE id = ?", (model_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def rename_model(self, model_id: int, new_name: str):
        """
        - Update the model's name and directory path in the database.
        - Synchronously rename both ./data/{old} and ./out/{old} directories.
        """
        info = self.get_model_basic_info(model_id)
        if not info:
            raise ValueError(f"Model {model_id} does not exist.")

        old_folder = os.path.basename(info["dir_path"])
        new_folder = f"{new_name}_{model_id}"

        old_out_abs = self._abs(info["dir_path"])
        new_out_abs = os.path.join("out", new_folder)
        if os.path.exists(old_out_abs):
            os.rename(old_out_abs, new_out_abs)

        old_data_abs = os.path.join("data", old_folder)
        new_data_abs = os.path.join("data", new_folder)
        if os.path.exists(old_data_abs):
            os.rename(old_data_abs, new_data_abs)

        cur = self.conn.cursor()
        cur.execute(
            "UPDATE models SET name = ?, dir_path = ? WHERE id = ?",
            (new_name, self._rel(new_out_abs), model_id)
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Training & Inference Configs, Logs & History
    # ------------------------------------------------------------------ #

    def save_training_config(self, model_id: int, cfg: dict):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO training_configs(model_id, config_json) "
            "VALUES(?,?) "
            "ON CONFLICT(model_id) DO UPDATE SET config_json=excluded.config_json",
            (model_id, json.dumps(cfg, ensure_ascii=False, indent=2))
        )
        self.conn.commit()

    def save_training_log(self, model_id: int, log_path: str):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO training_logs(model_id, log_path) "
            "VALUES(?,?) "
            "ON CONFLICT(model_id) DO UPDATE SET log_path=excluded.log_path",
            (model_id, self._rel(log_path))
        )
        self.conn.commit()

    def save_inference_config(self, model_id: int, cfg: dict):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO inference_configs(model_id, config_json) "
            "VALUES(?,?) "
            "ON CONFLICT(model_id) DO UPDATE SET config_json=excluded.config_json",
            (model_id, json.dumps(cfg, ensure_ascii=False, indent=2))
        )
        self.conn.commit()

    def save_inference_history(self, model_id: int, content: str):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO inference_history(model_id, content) "
            "VALUES(?,?) "
            "ON CONFLICT(model_id) DO UPDATE SET content=excluded.content",
            (model_id, content)
        )
        self.conn.commit()

    def clear_inference_history(self, model_id: int):
        """
        Delete all inference history entries for the given model.
        """
        self.conn.execute("DELETE FROM inference_history WHERE model_id = ?", (model_id,))
        self.conn.commit()

    def get_training_config(self, model_id: int):
        """
        Retrieve the training configuration JSON for the given model.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT config_json FROM training_configs WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return json.loads(row["config_json"]) if row else None

    def get_training_log_path(self, model_id: int):
        """
        Retrieve the absolute path to the training log file for the given model.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT log_path FROM training_logs WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return self._abs(row["log_path"]) if row else ""

    def get_inference_config(self, model_id: int):
        """
        Retrieve the inference configuration JSON for the given model.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT config_json FROM inference_configs WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return json.loads(row["config_json"]) if row else None

    def get_inference_history(self, model_id: int):
        """
        Retrieve the last inference history content for the given model.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT content FROM inference_history WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return row["content"] if row else ""

    # ------------------------------------------------------------------ #
    # Delete Model (cascade delete files & DB entries)
    # ------------------------------------------------------------------ #
    def delete_model(self, model_id: int):
        """
        Delete the model record and remove associated folders from disk.
        """
        info = self.get_model_basic_info(model_id)
        if info:
            folder = os.path.basename(info["dir_path"])
            for p in (os.path.join("data", folder), self._abs(info["dir_path"])):
                if os.path.exists(p):
                    shutil.rmtree(p, ignore_errors=True)

        self.conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # List All Models
    # ------------------------------------------------------------------ #
    def get_all_models(self):
        """
        Return a list of all models ordered by creation time descending.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id, name FROM models ORDER BY created_at DESC")
        return [dict(row) for row in cur.fetchall()]
