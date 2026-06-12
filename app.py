"""
app.py

Main file for running the Multimodal Agent.
"""
import tools
from state import create_initial_state
from agent import build_agent


def main():

    files = [
        "examples/dashboard.png",
        "examples/context.txt"
    ]
    
    user_question = "What is the main problem shown in the files, and what should we do next?"

    state = create_initial_state(user_question, files)

    agent = build_agent()

    # recursion_limit caps total node visits; default=25 but happy-path hits ~24, so bump to 100
    final_state = agent.invoke(state, config={"recursion_limit": 100})

    final_answer = tools.format_final_output(final_state)

    tools.save_output_to_file(final_state)

    print(final_answer)


if __name__ == "__main__":
    main()
