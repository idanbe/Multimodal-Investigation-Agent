"""
app.py

Main file for running the Multimodal Agent.
"""
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

    final_state = agent.invoke(state)  # Invoke the agent

    print(f"Final state: {final_state}")


if __name__ == "__main__":
    print("App.py is running")
    main()
