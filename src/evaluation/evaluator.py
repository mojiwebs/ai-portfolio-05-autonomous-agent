import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List
from openai import OpenAI

# Initialize OpenAI client for local Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
MODEL_NAME = "llama3.2"

# Import tools so the evaluator can test direct execution or inspect behaviors
from src.agent.tools import TOOL_DEFINITIONS, TOOL_REGISTRY


@dataclass
class TestCase:
    id: str
    goal: str
    expected_tool: str
    must_contain_keyword: str


# --- Golden Dataset (Evaluation Test Cases) ---
GOLDEN_DATASET: List[TestCase] = [
    TestCase(
        id="TC-01",
        goal="Check our system database for failed audit logs.",
        expected_tool="query_audit_logs",
        must_contain_keyword="safety_manual.pdf",
    ),
    TestCase(
        id="TC-02",
        goal="Calculate what 1500 * 0.21 is for our tax report.",
        expected_tool="calculate_expression",
        must_contain_keyword="315",
    ),
]


async def run_eval_for_test_case(test_case: TestCase) -> Dict[str, Any]:
    """Runs a single test case through the agent loop and evaluates success."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous enterprise AI agent. Use your tools when"
                " necessary to fulfill the user's goal."
            ),
        },
        {"role": "user", "content": test_case.goal},
    ]

    tools_called = []
    final_response = ""

    # Run agent steps
    for _ in range(5):
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=TOOL_DEFINITIONS, temperature=0.0
        )
        response_message = response.choices[0].message
        messages.append(response_message)

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                tools_called.append(func_name)

                # Execute tool
                tool_func = TOOL_REGISTRY[func_name]
                tool_result = tool_func(**func_args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )
        else:
            final_response = response_message.content or ""
            break

    # --- Evaluation Scoring Logic ---
    tool_passed = test_case.expected_tool in tools_called
    keyword_passed = test_case.must_contain_keyword.lower() in final_response.lower()
    overall_passed = tool_passed and keyword_passed

    return {
        "test_id": test_case.id,
        "goal": test_case.goal,
        "tools_called": tools_called,
        "expected_tool": test_case.expected_tool,
        "tool_passed": tool_passed,
        "final_response": final_response,
        "keyword_passed": keyword_passed,
        "overall_passed": overall_passed,
    }


async def run_evaluation_suite():
    """Executes the complete evaluation harness across all golden test cases."""
    print("\n" + "=" * 50)
    print(" RUNNING ENTERPRISE AGENT EVALUATION HARNESS ")
    print("=" * 50)

    results = []
    for test_case in GOLDEN_DATASET:
        print(f"\n[Evaluating] {test_case.id}: {test_case.goal}")
        res = await run_eval_for_test_case(test_case)
        results.append(res)

        print(f"  -> Tool Match ({test_case.expected_tool}): {res['tool_passed']}")
        print(
            f"  -> Keyword Match ('{test_case.must_contain_keyword}'): {res['keyword_passed']}"
        )
        print(f"  -> Test Result: {'✅ PASS' if res['overall_passed'] else '❌ FAIL'}")

    # Summary Report
    passed_count = sum(1 for r in results if r["overall_passed"])
    total_count = len(results)
    score_percentage = (passed_count / total_count) * 100

    print("\n" + "=" * 50)
    print(
        f" EVALUATION SUMMARY: {passed_count}/{total_count} Passed"
        f" ({score_percentage:.1f}%)"
    )
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(run_evaluation_suite())
