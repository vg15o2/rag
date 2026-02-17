import os
from dotenv import load_dotenv

load_dotenv()


def main():
    # Validate API key before initializing
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-api-key-here":
        print("Error: Please set your OPENAI_API_KEY in the .env file.")
        return

    print("Initializing Crypto Data Analyst...")
    from agent import create_agent

    agent = create_agent()

    print("\n=== Crypto Data Analyst ===")
    print("Ask me anything about cryptocurrency price data (2013-2021).")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("cya!")
            break

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                {"recursion_limit": 8},
            )
            # Extract the final AI message
            ai_message = result["messages"][-1]
            print(f"\nAnalyst: {ai_message.content}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
