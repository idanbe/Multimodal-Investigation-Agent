"""
app.py

Main file for running the Multimodal Agent.
"""
import config
import tools
from state import create_initial_state
from agent import build_agent


def main():

    files = [
        "examples/dashboard.png",
        "examples/context.txt"
    ]

    user_question = "What caused home prices to drop after 2007, and what does the chart show about the severity of that decline?"

    state = create_initial_state(user_question, files)

    if config.use_mocks():
        mode = "MOCK"
    else:
        mode = ("REAL via OpenRouter:\n"
                f"image={config.openrouter_model('image')},\n"
                f"document={config.openrouter_model('document')},\n"
                f"audio={config.openrouter_model('audio')},\n"
                f"answer={config.openrouter_model('answer')}")
    print(f"Model mode: {mode}\n")

    agent = build_agent()

    # recursion_limit caps total node visits; default=25 but happy-path hits ~24, so bump to 100
    final_state = agent.invoke(state, config={"recursion_limit": 100})

    final_answer = tools.format_final_output(final_state)

    tools.save_output_to_file(final_state)

    print(final_answer)


if __name__ == "__main__":
    main()
