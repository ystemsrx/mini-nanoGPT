# app.py — Project entry point
"""
# Mini-NanoGPT

Based on karpathy/nanoGPT with a GUI and extended features that make GPT model training intuitive and accessible.

- 🚀 One-click data processing, training and inference
- 🎨 Real-time training visualization and logging
- 🔧 Character-level and custom tokenizer support (Qwen, etc.)
- 💾 Checkpoint resume and model evaluation
- 🌏 Multi-language interface (English/Chinese)
- 📊 Rich learning rate scheduling options
- 🎛️ Visual configuration for all parameters, no code editing needed

Compared to the original nanoGPT, this project adds more practical features and flexible configuration options, giving you complete control over the training process. Whether for learning or experimentation, it helps you explore GPT models more easily.

Built with PyTorch and Gradio, featuring clean code structure, perfect for deep learning beginners and researchers.
"""

from src.ui import build_app_interface

if __name__ == "__main__":
    # Build and launch the Gradio app
    demo = build_app_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
