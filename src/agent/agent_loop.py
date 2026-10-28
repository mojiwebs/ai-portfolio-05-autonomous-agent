import json
import os
from openai import OpenAI
from src.agent.tools import TOOL_DEFINITIONS, TOOL_REGISTRY

# Initialize OpenAI client pointed to local Ollama instance
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Required by SDK, but ignored by local Ollama
)

MODEL_NAME = "llama3.2"


def run_autonomous_agent(user_goal: str):
    """Executes an autonomous ReAct loop using Llama 3.2 and local tool registry."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous enterprise AI agent. You have access to"
                " specific tools. Always analyze the user's goal, invoke tools"
                " when needed to gather data or perform calculations, and"
                " synthesize a final, clear answer."
            ),
        },
        {"role": "user", "content": user_goal},
    ]

    print(f"\n[Agent Goal]: {user_goal}")
    print("=" * 60)

    # Run agent iteration loop (Max 5 steps to prevent infinite execution)
    for step in range(5):
        print(f"\n-> Agent Iteration {step + 1}: Thinking...")

        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=TOOL_DEFINITIONS, temperature=0.0
        )

        response_message = response.choices[0].message

        # Append assistant's response to history
        messages.append(response_message)

        # Check if the model requested any tool executions
        if response_message.tool_calls:
            print(
                f"[Action Required]: Model requested"
                f" {len(response_message.tool_calls)} tool call(s)."
            )

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(
                    f"   -> Executing Tool: `{function_name}` with arguments:"
                    f" {function_args}"
                )

                # Lookup and execute tool from our registry
                if function_name in TOOL_REGISTRY:
                    tool_function = TOOL_REGISTRY[function_name]
                    tool_result = tool_function(**function_args)
                else:
                    tool_result = json.dumps(
                        {"error": f"Tool {function_name} not found."}
                    )

                print(f"   <- Tool Result: {tool_result}")

                # Append tool output back to message history
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )

        else:
            # No tool call requested; model provided final text answer
            print("\n" + "=" * 20 + " FINAL AGENT CONCLUSION " + "=" * 20)
            print(response_message.content)
            return response_message.content

    print("\n[Warning]: Agent reached maximum step limit without concluding.")
    return "Agent reached maximum iterations."


if __name__ == "__main__":
    goal = (
        "Check our database for failed audit logs and calculate what 1500 * 0.21"
        " is based on those records."
    )
    run_autonomous_agent(goal)
