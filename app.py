"""
app.py

Main file for running the Multimodal Investigation Agent.
"""
import os

import gradio as gr

import config
import tools
from state import create_initial_state
from agent import build_agent


SUPPORTED_EXTENSIONS = {
    ext
    for exts in tools.MODALITY_TO_EXTENSION_MAP.values()
    for ext in exts
}

DEFAULT_QUESTION = (
    "What caused home prices to drop after 2007, and what does the "
    "chart show about the severity of that decline?"
)


def load_example_files(folder: str = "examples") -> list[str]:
    """Return sorted paths of every supported file in `folder`."""
    paths = []
    for name in sorted(os.listdir(folder)):
        if name.startswith("."):
            continue
        ext = name.rsplit(".", 1)[-1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            paths.append(os.path.join(folder, name))
    return paths


def run_agent(user_question: str, files: list[str]) -> str:
    """Run the agent for a question over files; return formatted answer text."""
    state = create_initial_state(user_question, files)
    agent = build_agent()
    # recursion_limit caps node visits; happy-path hits ~24, default=25, bump to 100
    final_state = agent.invoke(state, config={"recursion_limit": 100})
    tools.save_output_to_file(final_state)
    return tools.format_final_output(final_state)


def main():
    files = load_example_files()
    user_question = DEFAULT_QUESTION

    if config.use_mocks():
        mode = "MOCK"
    else:
        mode = ("REAL via OpenRouter:\n"
                f"image={config.openrouter_model('image')},\n"
                f"document={config.openrouter_model('document')},\n"
                f"audio={config.openrouter_model('audio')},\n"
                f"answer={config.openrouter_model('answer')}")
    print(f"Model mode: {mode}\n")

    print(run_agent(user_question, files))


def launch_ui():
    """Launch the Gradio UI. Layout depends on mock vs real mode."""
    files = load_example_files()
    mock_mode = config.use_mocks()
    mode_label = "MOCK" if mock_mode else "REAL (OpenRouter)"

    def on_run(question: str) -> str:
        question = (question or "").strip()
        if not question:
            return "Please enter a question."
        return run_agent(question, files)

    with gr.Blocks(title="Multimodal Investigation Agent") as demo:
        gr.Markdown(
            f"# Multimodal Investigation Agent\n**Mode:** {mode_label}")
        gr.Markdown("**Files:** " + ", ".join(files))

        question_box = gr.Textbox(
            label="User question",
            value=DEFAULT_QUESTION if mock_mode else "",
            interactive=not mock_mode,
            lines=2,
        )
        run_btn = gr.Button("Run agent", variant="primary")
        answer_box = gr.Textbox(label="Answer", lines=20)

        run_btn.click(fn=on_run, inputs=question_box, outputs=answer_box)

    demo.launch()


if __name__ == "__main__":
    main()
