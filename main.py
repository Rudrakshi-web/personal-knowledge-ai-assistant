print("🔥 MAIN.PY IS RUNNING")

from ingestion import run_ingestion
from rag_pipeline import ask_assistant

# 👇 THIS LINE BUILDS VECTOR DB FROM YOUR BOOK
run_ingestion("data")

print("🧠 Personal Knowledge AI Assistant Ready!")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Assistant: Goodbye! 👋")
        break

    response = ask_assistant(user_input)
    print("\nAssistant:", response, "\n")
