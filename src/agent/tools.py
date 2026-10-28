import json
import math
from typing import Any, Dict, List


# --- 1. Concrete Tool Implementation Functions ---


def calculate_expression(expression: str) -> str:
    """Evaluates a mathematical expression securely."""
    try:
        allowed_names = {"sqrt": math.sqrt, "pi": math.pi, "e": math.e}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


# get system health
def get_current_system_health() -> str:
    """Simulates a system health check, returning a JSON string with status and metrics."""
    health_status = {
        "status": "OK",
        "uptime": "72 hours",
        "cpu_usage": "45%",
        "memory_usage": "68%",
        "disk_space": "120GB free of 500GB",
    }
    return json.dumps(health_status)


def query_audit_logs(status_filter: str = "ALL") -> str:
    """Queries the system database for recent document audit logs and statuses."""
    # Simulated enterprise database records (tying into our Project 4 data layer)
    mock_records = [
        {
            "document_name": "safety_manual.pdf",
            "overall_status": "FAIL",
            "total_chunks": 14,
        },
        {
            "document_name": "compliance_policy.pdf",
            "overall_status": "PASS",
            "total_chunks": 8,
        },
        {
            "document_name": "architecture_spec.pdf",
            "overall_status": "PASS",
            "total_chunks": 22,
        },
    ]

    if status_filter != "ALL":
        filtered = [
            r for r in mock_records if r["overall_status"] == status_filter.upper()
        ]
        return json.dumps(filtered)

    return json.dumps(mock_records)


# --- 2. OpenAI / Ollama-Compatible Tool Schemas ---

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": "Evaluate a mathematical expression. Use this for any numeric or financial calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '1500 * 0.21').",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_system_health",
            "description": "Retrieve the current health status of the system.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_audit_logs",
            "description": "Query the system database for document audit logs, compliance statuses, and chunk metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status_filter": {
                        "type": "string",
                        "description": "Filter records by status: 'PASS', 'FAIL', or 'ALL'.",
                        "default": "ALL",
                    }
                },
                "required": [],
            },
        },
    },
]


# --- 3. Executable Function Mapping Registry ---

TOOL_REGISTRY: Dict[str, callable] = {
    "calculate_expression": calculate_expression,
    "query_audit_logs": query_audit_logs,
    "get_current_system_health": get_current_system_health,
}
