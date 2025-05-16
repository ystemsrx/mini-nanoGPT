[English](README.md) | [简体中文](README.zh.md)

# Mini-NanoGPT - `database` Branch

## 📌 Project Overview

**Mini-NanoGPT** is a lightweight visual training platform based on Karpathy’s [nanoGPT](https://github.com/karpathy/nanoGPT), designed to help deep learning beginners, researchers, and developers quickly grasp the GPT model training workflow through an intuitive graphical interface.

This `database` branch builds upon the original features by introducing **model management database support**, enabling unified storage and tracking of model metadata, training configurations, inference parameters, and execution history — greatly enhancing systematic and scalable experiment management.

---

## 🚀 Branch Highlights: Database Features

### ✅ Model Registration & Persistent Tracking

* Introduces an **SQLite database**. A `DBManager` component automatically assigns a unique ID to each new model and records metadata such as model name, creation time, and file path.
* Centralized management of all model metadata, enabling easy lookup and state maintenance.

### ✅ Configuration, Execution, and Files

* Automatically stores **hyperparameter configurations**, **log paths**, **inference parameters**, and **generation history** during training.
* Supports parameter rollback, experiment resumption, and auto-filled UI forms for reproducible training.
* Automatically creates structured directories (e.g., `out/{model_name}_{model_id}`, `data/{model_name}_{model_id}`).

### ✅ Visual Model Management Interface

* A new **Model Management** tab added to the frontend:

  * Visual browsing of all models (Name + ID)
  * Supports adding, switching, refreshing, and deleting models — all operations are fully synchronized with the database.
* All frontend actions use `DBManager` to maintain consistency between backend data and UI state.

> ⚠️ Note: This branch only adds model database and management functionality. The training and inference processes remain unchanged — see the main branch documentation for details.

---

## 🧪 Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Model registration, configuration storage, and related operations during training and inference are implemented in the `DBManager` source file.

---

## 📁 File Structure Overview

### `db_manager.py` (Core Module)

Encapsulates the `DBManager` class, responsible for:

* Initializing the SQLite database and creating:

  * Model metadata table
  * Training configuration table
  * Log path table
  * Inference parameters and history table
* Providing core methods such as:

  * `register_model`: Register a new model
  * `rename_model`: Rename a model and update the corresponding folder
  * `delete_model`: Delete a model and related directories
  * `get_model_basic_info` / `get_all_models`: Query model information
  * Save/load training and inference configs, history records, etc.

This module ensures consistency between database records and the file system, reducing manual maintenance and improving reliability.

---

### `main.py` (Main Entry Point)

Integrates the database management logic:

* Embeds `DBManager` API calls into data loading, model training, and inference stages to automate model registration and configuration storage.
* The frontend adds a **“Model Management”** tab using Gradio to enable interactive database operations for models.
