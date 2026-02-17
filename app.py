import os
from dotenv import load_dotenv

load_dotenv()


def main():
    # Validate API key before initializing
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-api-key-here":
        print("Error: Please set your OPENAI_API_KEY in the .env file.")
        return

    print("Initializing Crypto Data Analyst (RAG mode)...")
    from rag_agent import create_rag_agent

    agent, reset_repl = create_rag_agent()

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
            reset_repl()
            final_content = ""
            for step in agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                {"recursion_limit": 25},
                stream_mode="updates",
            ):
                for node_output in step.values():
                    for msg in node_output.get("messages", []):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            names = [tc["name"] for tc in msg.tool_calls]
                            print(f"  thinking... ({', '.join(names)})")
                        elif msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
                            final_content = msg.content

            print(f"\n{final_content or 'Could not generate an answer.'}\n")

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
