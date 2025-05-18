# app.py — Project entry point

from src.ui import build_app_interface

if __name__ == "__main__":
    # Build and launch the Gradio app
    demo = build_app_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
