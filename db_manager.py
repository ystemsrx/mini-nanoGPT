# db_manager.py

import os, sqlite3, json, time, shutil
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.getcwd())  # 项目根目录（绝对）

class DBManager:
    """
    统一的 SQLite 管理器：
      · register_model            – dir_path 可选，若缺省则自动 ./out/{name}_{id}/
      · rename_model              – 同步重命名 ./data 与 ./out 目录
      · delete_model              – 同步删除 ./data 与 ./out 目录
      · get_model_basic_info      – 返回 {'name','dir_path'}
    """
    # ------------------------------------------------------------------ #
    # 初始化 & 建表
    # ------------------------------------------------------------------ #
    def __init__(self, db_path: str = "model_registry.db"):
        rel_db_path = os.path.relpath(db_path, PROJECT_ROOT)
        self.conn = sqlite3.connect(rel_db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
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
    # 内部路径工具
    # ------------------------------------------------------------------ #
    def _rel(self, path: str) -> str:
        return os.path.relpath(path, PROJECT_ROOT)

    def _abs(self, rel_path: str) -> str:
        return os.path.join(PROJECT_ROOT, rel_path) if rel_path else ""

    # ------------------------------------------------------------------ #
    # 模型注册 / 基础信息
    # ------------------------------------------------------------------ #
    def register_model(self, name: str, dir_path: Optional[str] = None) -> int:
        """
        若 dir_path=None，则自动创建 ./out/{name}_{id}/ 并持久化相对路径。
        返回模型唯一 id（毫秒级时间戳）。
        """
        cur = self.conn.cursor()

        if dir_path:                               # 若提供目录，检查是否已存在
            rel = self._rel(dir_path)
            cur.execute("SELECT id FROM models WHERE dir_path = ?", (rel,))
            row = cur.fetchone()
            if row:
                return row["id"]
            dir_path_rel = rel
        else:                                      # 自动生成目录
            ts_ms_tmp = int(time.time() * 1000)
            auto_folder = f"{name}_{ts_ms_tmp}"
            dir_path_rel = self._rel(os.path.join("out", auto_folder))
            os.makedirs(self._abs(dir_path_rel), exist_ok=True)

        ts_ms = int(time.time() * 1000)
        cur.execute(
            "INSERT INTO models(id, name, dir_path, created_at) VALUES(?,?,?,datetime('now'))",
            (ts_ms, name, dir_path_rel)
        )
        self.conn.commit()
        return ts_ms
    
    def get_model_id_by_dir(self, dir_path: str) -> Optional[int]:
        dir_path_rel = self._rel(dir_path)
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM models WHERE dir_path = ?", (dir_path_rel,))
        row = cur.fetchone()
        return row["id"] if row else None
    
    def get_model_basic_info(self, model_id: int) -> Optional[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT name, dir_path FROM models WHERE id = ?", (model_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def rename_model(self, model_id: int, new_name: str):
        """
        · 更新数据库 name / dir_path
        · 同步重命名 ./data/{old} 与 ./out/{old}
        """
        info = self.get_model_basic_info(model_id)
        if not info:
            raise ValueError(f"Model {model_id} 不存在。")

        old_folder = os.path.basename(info["dir_path"])           # oldName_id
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
    # 训练 / 推理 配置 & 日志 & 历史（未改动，保持原实现）
    #   ↓↓↓  —— 以下方法完整保留，无任何修改 —— ↓↓↓
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
        self.conn.execute("DELETE FROM inference_history WHERE model_id = ?", (model_id,))
        self.conn.commit()

    def get_training_config(self, model_id: int):
        cur = self.conn.cursor()
        cur.execute("SELECT config_json FROM training_configs WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return json.loads(row["config_json"]) if row else None

    def get_training_log_path(self, model_id: int):
        cur = self.conn.cursor()
        cur.execute("SELECT log_path FROM training_logs WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return self._abs(row["log_path"]) if row else ""

    def get_inference_config(self, model_id: int):
        cur = self.conn.cursor()
        cur.execute("SELECT config_json FROM inference_configs WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return json.loads(row["config_json"]) if row else None

    def get_inference_history(self, model_id: int):
        cur = self.conn.cursor()
        cur.execute("SELECT content FROM inference_history WHERE model_id = ?", (model_id,))
        row = cur.fetchone()
        return row["content"] if row else ""

    # ------------------------------------------------------------------ #
    # 删除模型（级联删文件 + DB）
    # ------------------------------------------------------------------ #
    def delete_model(self, model_id: int):
        info = self.get_model_basic_info(model_id)
        if info:
            folder = os.path.basename(info["dir_path"])          # name_id
            for p in (os.path.join("data", folder), self._abs(info["dir_path"])):
                if os.path.exists(p):
                    shutil.rmtree(p, ignore_errors=True)

        self.conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # 列表
    # ------------------------------------------------------------------ #
    def get_all_models(self):
        cur = self.conn.cursor()
        cur.execute("SELECT id, name FROM models ORDER BY created_at DESC")
        return [dict(row) for row in cur.fetchall()]
