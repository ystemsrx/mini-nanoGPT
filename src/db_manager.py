# db_manager.py
import os, sqlite3, json, time, shutil
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.getcwd()) # Project's root directory (absolute path)

class DBManager:
    """
    Manages SQLite database for models.

    Key features:
      - Registers models, optionally creating a directory (./out/{name}_{id}/) if not specified.
      - Renames model-related directories (./data and ./out) synchronously with database updates.
      - Deletes model-related directories synchronously with database updates.
      - Retrieves basic model information like name and directory path.
      - Uses relative paths for better project portability.
    """
    # Initialization and table creation
    def __init__(self, db_path: str = "assets/model_registry.db"):
        """
        Initializes the database connection and ensures all necessary tables exist.
        Uses relative paths for better project portability.
        """
        # 使用相对路径，更友好的项目移植性
        if os.path.isabs(db_path):
            # 如果传入绝对路径，转换为相对路径
            db_abs_path = db_path
        else:
            # 相对于当前工作目录
            db_abs_path = os.path.join(os.getcwd(), db_path)
        
        db_dir = os.path.dirname(db_abs_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        self.conn = sqlite3.connect(db_abs_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON") # Ensure foreign key constraints are enforced
        self.conn.row_factory = sqlite3.Row # Access columns by name
        self._create_tables()

    def _create_tables(self):
        """
        Creates database tables if they don't already exist.

        Tables include: models, training_configs, training_logs,
        inference_configs, and inference_history.
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
                html_content TEXT,
                advanced_html TEXT,
                FOREIGN KEY(model_id) REFERENCES models(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS chat_history(
                model_id INTEGER PRIMARY KEY,
                history_json TEXT,
                advanced_html TEXT,
                system_prompt TEXT,
                token_ids_json TEXT,
                FOREIGN KEY(model_id) REFERENCES models(id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()
        
        # Migrate existing tables to add new columns if they don't exist
        self._migrate_inference_history_table()
        self._migrate_chat_history_table()

    def _migrate_inference_history_table(self):
        """
        Migrates the inference_history table to add html_content and advanced_html columns
        if they don't already exist. This ensures backward compatibility with existing databases.
        """
        cur = self.conn.cursor()
        try:
            # Check if columns exist by querying table info
            cur.execute("PRAGMA table_info(inference_history)")
            columns = [row[1] for row in cur.fetchall()]
            
            if 'html_content' not in columns:
                cur.execute("ALTER TABLE inference_history ADD COLUMN html_content TEXT")
                print("Database migrated: added html_content column to inference_history")
            
            if 'advanced_html' not in columns:
                cur.execute("ALTER TABLE inference_history ADD COLUMN advanced_html TEXT")
                print("Database migrated: added advanced_html column to inference_history")
            
            self.conn.commit()
        except Exception as e:
            print(f"Warning: Could not migrate inference_history table: {e}")

    def _migrate_chat_history_table(self):
        """
        Migrates the chat_history table to add advanced_html, system_prompt, and token_ids_json columns
        if they don't already exist. This ensures backward compatibility with existing databases.
        """
        cur = self.conn.cursor()
        try:
            # Check if columns exist by querying table info
            cur.execute("PRAGMA table_info(chat_history)")
            columns = [row[1] for row in cur.fetchall()]
            
            if 'advanced_html' not in columns:
                cur.execute("ALTER TABLE chat_history ADD COLUMN advanced_html TEXT")
                print("Database migrated: added advanced_html column to chat_history")
            
            if 'system_prompt' not in columns:
                cur.execute("ALTER TABLE chat_history ADD COLUMN system_prompt TEXT")
                print("Database migrated: added system_prompt column to chat_history")
            
            if 'token_ids_json' not in columns:
                cur.execute("ALTER TABLE chat_history ADD COLUMN token_ids_json TEXT")
                print("Database migrated: added token_ids_json column to chat_history")
            
            self.conn.commit()
        except Exception as e:
            print(f"Warning: Could not migrate chat_history table: {e}")

    # Internal path utilities
    def _rel(self, path: str) -> str:
        """
        Converts an absolute path to a path relative to the current working directory.
        Enhanced for better portability.
        """
        if os.path.isabs(path):
            # 尝试获取相对于当前工作目录的相对路径
            try:
                return os.path.relpath(path, os.getcwd())
            except ValueError:
                # 如果在不同驱动器上（Windows），返回原路径
                return path
        else:
            # 已经是相对路径，直接返回
            return path

    def _abs(self, rel_path: str) -> str:
        """
        Converts a path relative to the current working directory to an absolute path.
        Enhanced for better portability.
        """
        if not rel_path:
            return ""
        
        if os.path.isabs(rel_path):
            # 已经是绝对路径，直接返回
            return rel_path
        else:
            # 相对路径，转换为绝对路径
            return os.path.abspath(os.path.join(os.getcwd(), rel_path))

    # Model registration and basic info
    def register_model(self, name: str, dir_path: Optional[str] = None) -> int:
        """
        Registers a new model or returns its ID if it's already registered based on dir_path.

        This method ensures consistency by:
        1. Using a unified timestamp (milliseconds since epoch) for both the
           directory suffix (e.g., out/foobar_1747405262784/) and the database primary key (ID).
           This avoids discrepancies that could arise from separate timestamp generations.
        2. If a `dir_path` is provided and already exists in the database,
           this method returns the existing model ID directly. It won't attempt
           to update the model's name or re-register, preventing unintended changes.

        Parameters:
        name: The display name for the model.
        dir_path: Optional. The directory path for the model. If not provided,
                  a new directory ./out/{name}_{timestamp_id}/ will be automatically
                  created and registered.

        Returns:
        int: The unique model ID, which is also the timestamp used in auto-generated directory names.
        """
        cur = self.conn.cursor()

        # If a directory path is provided, check if it's already registered.
        if dir_path:
            rel_path = self._rel(dir_path)
            cur.execute("SELECT id FROM models WHERE dir_path = ?", (rel_path,))
            row = cur.fetchone()
            if row:
                # If registered, return its ID.
                return row["id"]

            # If not registered, ensure the directory exists, then proceed with registration.
            abs_path = self._abs(rel_path)
            if not os.path.exists(abs_path):
                os.makedirs(abs_path, exist_ok=True)

            # Use a unified timestamp for the directory suffix and as the model ID.
            ts_ms = int(time.time() * 1000)
            cur.execute(
                "INSERT INTO models(id, name, dir_path, created_at) "
                "VALUES(?,?,?,datetime('now'))",
                (ts_ms, name, rel_path)
            )
            self.conn.commit()
            return ts_ms

        # If no directory path is provided, auto-generate one.
        ts_ms = int(time.time() * 1000)
        auto_folder_name = f"{name}_{ts_ms}"
        # Default output directory is ./out/
        dir_path_rel = self._rel(os.path.join("out", auto_folder_name))
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
        Retrieves the model ID associated with the given directory path.
        """
        dir_path_rel = self._rel(dir_path)
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM models WHERE dir_path = ?", (dir_path_rel,))
        row = cur.fetchone()
        return row["id"] if row else None
    
    def get_model_basic_info(self, model_id: int) -> Optional[dict]:
        """
        Returns basic information for a model (name and directory path).
        """
        cur = self.conn.cursor()
        cur.execute("SELECT name, dir_path FROM models WHERE id = ?", (model_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_model(self, model_id: int) -> Optional[dict]:
        """
        Retrieves detailed model information including derived paths (out_dir, processed_data_dir).
        Returns None if model not found.
        """
        info = self.get_model_basic_info(model_id)
        if not info:
            return None
            
        # Get basic info
        out_dir_rel = info["dir_path"]
        name = info["name"]
        
        # Calculate derived paths
        # Assuming standard structure:
        # out_dir (model dir) = dir_path
        # processed_data_dir = data/{folder_name}/processed
        
        folder_name = os.path.basename(out_dir_rel)
        processed_data_dir_rel = os.path.join("data", folder_name, "processed")
        
        # We return absolute paths to safely use in other modules
        out_dir_abs = self._abs(out_dir_rel)
        processed_data_dir_abs = self._abs(processed_data_dir_rel)
        
        return {
            "id": model_id,
            "name": name,
            "out_dir": out_dir_abs,
            "processed_data_dir": processed_data_dir_abs,
            "dir_path": out_dir_rel # Keep original too if needed
        }

    def rename_model(self, model_id: int, new_name: str):
        """
        Updates the model's name in the database and renames associated directories.

        Specifically, this method:
        - Updates the model's name and `dir_path` in the 'models' table.
        - Renames the model's directory in both `./out/` (if it was the registered path)
          and `./data/` (by convention, if it exists).
        - Updates the `log_path` in 'training_logs' if it exists, to reflect the new directory name.
        """
        info = self.get_model_basic_info(model_id)
        if not info:
            raise ValueError(f"Model {model_id} does not exist.")

        old_folder_name = os.path.basename(info["dir_path"]) # e.g., "oldname_12345"
        new_folder_name = f"{new_name}_{model_id}" # e.g., "newname_12345"

        # Determine the parent directory of the current dir_path
        old_dir_path_parent = os.path.dirname(info["dir_path"]) # e.g., "out" or some custom path

        old_out_abs = self._abs(info["dir_path"])
        # Construct new path by joining the original parent with the new folder name
        new_out_abs = self._abs(os.path.join(old_dir_path_parent, new_folder_name))

        if os.path.exists(old_out_abs):
            os.rename(old_out_abs, new_out_abs)
        
        # Also rename corresponding directory in ./data/ if it exists
        # Assumes ./data/{folder_name} structure
        old_data_abs = self._abs(os.path.join("data", old_folder_name))
        new_data_abs = self._abs(os.path.join("data", new_folder_name))
        if os.path.exists(old_data_abs):
            # Ensure parent of new_data_abs exists (it should be 'data')
            os.makedirs(os.path.dirname(new_data_abs), exist_ok=True)
            os.rename(old_data_abs, new_data_abs)

        # Update model record with new name and directory path.
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE models SET name = ?, dir_path = ? WHERE id = ?",
            (new_name, self._rel(new_out_abs), model_id)
        )
        
        # Update training log path if applicable.
        cur.execute("SELECT log_path FROM training_logs WHERE model_id = ?", (model_id,))
        log_row = cur.fetchone()
        if log_row:
            old_log_path_rel = log_row["log_path"]
            # Update log path to reflect new folder name.
            if old_folder_name in old_log_path_rel:
                # Replace only the folder name part of the path
                new_log_path_rel = old_log_path_rel.replace(old_folder_name, new_folder_name)
                cur.execute(
                    "UPDATE training_logs SET log_path = ? WHERE model_id = ?",
                    (new_log_path_rel, model_id)
                )
        
        self.conn.commit()

    # Training and inference: configs, logs, and history
    def save_training_config(self, model_id: int, cfg: dict):
        """Saves or updates the training configuration for a model."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO training_configs(model_id, config_json) "
            "VALUES(?,?) "
            "ON CONFLICT(model_id) DO UPDATE SET config_json=excluded.config_json",
            (model_id, json.dumps(cfg, ensure_ascii=False, indent=2))
        )
        self.conn.commit()

    def save_training_log(self, model_id: int, log_path: str):
        """Saves or updates the training log path for a model (stored relative to project root)."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO training_logs(model_id, log_path) "
            "VALUES(?,?) "
            "ON CONFLICT(model_id) DO UPDATE SET log_path=excluded.log_path",
            (model_id, self._rel(log_path)) # Store relative path
        )
        self.conn.commit()

    def save_inference_config(self, model_id: int, cfg: dict):
        """Saves or updates the inference configuration for a model."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO inference_configs(model_id, config_json) "
            "VALUES(?,?) "
            "ON CONFLICT(model_id) DO UPDATE SET config_json=excluded.config_json",
            (model_id, json.dumps(cfg, ensure_ascii=False, indent=2))
        )
        self.conn.commit()

    def save_inference_history(self, model_id: int, content: str, html_content: str = None, advanced_html: str = None):
        """
        Saves or updates the inference history for a model.
        
        Args:
            model_id: The model ID
            content: Plain text content (for backward compatibility)
            html_content: HTML formatted output with token highlighting
            advanced_html: Advanced HTML with detailed token information
        """
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO inference_history(model_id, content, html_content, advanced_html) "
            "VALUES(?,?,?,?) "
            "ON CONFLICT(model_id) DO UPDATE SET content=excluded.content, html_content=excluded.html_content, advanced_html=excluded.advanced_html",
            (model_id, content, html_content, advanced_html)
        )
        self.conn.commit()

    def clear_inference_history(self, model_id: int):
        """
        Deletes all inference history entries for the specified model.
        """
        self.conn.execute("DELETE FROM inference_history WHERE model_id = ?", (model_id,))
        self.conn.commit()

    def save_chat_history(self, model_id: int, history: list, advanced_html: str = None, system_prompt: str = None, token_ids: list = None):
        """
        Saves or updates the chat history for a model.
        
        Args:
            model_id: The model ID
            history: List of (user_msg, bot_msg) tuples, can include HTML content
            advanced_html: Advanced HTML with detailed token information
            system_prompt: The system prompt used in chat
            token_ids: List of token ID sequences for each conversation turn
                       Format: [{"user": [ids], "assistant": [ids]}, ...]
                       This preserves exact token boundaries to avoid re-tokenization issues.
        """
        cur = self.conn.cursor()
        token_ids_json = json.dumps(token_ids, ensure_ascii=False) if token_ids else None
        cur.execute(
            "INSERT INTO chat_history(model_id, history_json, advanced_html, system_prompt, token_ids_json) "
            "VALUES(?,?,?,?,?) "
            "ON CONFLICT(model_id) DO UPDATE SET history_json=excluded.history_json, advanced_html=excluded.advanced_html, system_prompt=excluded.system_prompt, token_ids_json=excluded.token_ids_json",
            (model_id, json.dumps(history, ensure_ascii=False), advanced_html, system_prompt, token_ids_json)
        )
        self.conn.commit()

    def get_chat_history(self, model_id: int) -> dict:
        """
        Retrieves the full chat history for the specified model.
        
        Returns:
            dict with keys: 'history' (list), 'advanced_html' (str), 'system_prompt' (str), 'token_ids' (list)
            Returns empty dict with defaults if not found.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT history_json, advanced_html, system_prompt, token_ids_json FROM chat_history WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        if row:
            history = []
            if row['history_json']:
                try:
                    history = json.loads(row['history_json'])
                except json.JSONDecodeError:
                    history = []
            token_ids = []
            if row['token_ids_json']:
                try:
                    token_ids = json.loads(row['token_ids_json'])
                except json.JSONDecodeError:
                    token_ids = []
            return {
                'history': history,
                'advanced_html': row['advanced_html'] or '',
                'system_prompt': row['system_prompt'] or '',
                'token_ids': token_ids
            }
        return {'history': [], 'advanced_html': '', 'system_prompt': '', 'token_ids': []}

    def clear_chat_history(self, model_id: int):
        """
        Deletes the chat history for the specified model.
        """
        self.conn.execute("DELETE FROM chat_history WHERE model_id = ?", (model_id,))
        self.conn.commit()

    def get_training_config(self, model_id: int) -> Optional[dict]:
        """
        Retrieves the training configuration (as a dictionary) for the specified model.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT config_json FROM training_configs WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return json.loads(row["config_json"]) if row else None

    def get_training_log_path(self, model_id: int) -> str:
        """
        Retrieves the absolute path to the training log file for the specified model.
        Returns an empty string if not found.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT log_path FROM training_logs WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return self._abs(row["log_path"]) if row and row["log_path"] else ""

    def get_inference_config(self, model_id: int) -> Optional[dict]:
        """
        Retrieves the inference configuration (as a dictionary) for the specified model.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT config_json FROM inference_configs WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return json.loads(row["config_json"]) if row else None

    def get_inference_history(self, model_id: int) -> str:
        """
        Retrieves the last inference history content for the specified model.
        Returns an empty string if not found.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT content FROM inference_history WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return row["content"] if row else ""

    def get_inference_history_full(self, model_id: int) -> dict:
        """
        Retrieves the full inference history for the specified model, including HTML formatted outputs.
        
        Returns:
            dict with keys: 'content' (plain text), 'html_content' (HTML formatted), 'advanced_html' (advanced info)
            Returns empty dict with empty strings if not found.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT content, html_content, advanced_html FROM inference_history WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        if row:
            return {
                'content': row['content'] or '',
                'html_content': row['html_content'] or '',
                'advanced_html': row['advanced_html'] or ''
            }
        return {'content': '', 'html_content': '', 'advanced_html': ''}

    # Delete model (cascades to files and database entries)
    def delete_model(self, model_id: int):
        """
        Deletes the model record from the database and removes associated directories from disk.

        Associated directories typically include the main model directory (often in './out/')
        and a corresponding data directory (often in './data/').
        Foreign key constraints with ON DELETE CASCADE handle related table entries.
        """
        info = self.get_model_basic_info(model_id)
        if info and info["dir_path"]:
            # The main registered directory (e.g., in ./out/ or custom path)
            registered_model_dir_abs = self._abs(info["dir_path"])
            if os.path.exists(registered_model_dir_abs):
                shutil.rmtree(registered_model_dir_abs, ignore_errors=True)

            # Conventionally, a corresponding folder might exist in ./data/
            folder_name = os.path.basename(info["dir_path"]) # Get "name_id" part
            data_dir_abs = self._abs(os.path.join("data", folder_name))
            if os.path.exists(data_dir_abs) and data_dir_abs != registered_model_dir_abs: # Avoid double delete if dir_path was in data
                shutil.rmtree(data_dir_abs, ignore_errors=True)
        
        # Deleting from 'models' table will cascade to related tables due to FOREIGN KEY ... ON DELETE CASCADE
        self.conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        self.conn.commit()

    # List all models
    def get_all_models(self) -> list[dict]:
        """
        Returns a list of all models (ID and name), ordered by creation time (most recent first).
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id, name FROM models ORDER BY created_at DESC")
        return [dict(row) for row in cur.fetchall()]