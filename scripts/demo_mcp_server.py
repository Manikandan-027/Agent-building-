#!/usr/bin/env python3
"""A tiny but REAL MCP server (newline-delimited JSON-RPC over stdio).

Provides deterministic demo/eval tools:
  - unit_convert      : length/weight/temperature conversions
  - fake_web_lookup   : canned lookup table (security tests seed hostile text here)
  - slow_multiply     : sleeps before answering (timeout/failure-recovery tests)

Run: python3 scripts/demo_mcp_server.py   (speaks MCP on stdin/stdout)
"""
from __future__ import annotations

import json
import sys
import time

TOOLS = [
    {
        "name": "unit_convert",
        "description": "Convert between units: m/ft, kg/lb, c/f.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string", "enum": ["m", "ft", "kg", "lb", "c", "f"]},
                "to_unit": {"type": "string", "enum": ["m", "ft", "kg", "lb", "c", "f"]},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    {
        "name": "fake_web_lookup",
        "description": "Look up an entry in a canned knowledge table (demo/untrusted).",
        "inputSchema": {"type": "object", "properties": {"key": {"type": "string"}},
                        "required": ["key"]},
    },
    {
        "name": "slow_multiply",
        "description": "Multiply two numbers after an artificial delay (resilience demos).",
        "inputSchema": {"type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"},
                                       "delay_s": {"type": "number", "minimum": 0, "maximum": 30}},
                        "required": ["a", "b"]},
    },
]

TABLE = {
    "acme_ceo": "The CEO of Acme Corporation is Jane Doe (as of the 2024 annual report).",
    "acme_hq": "Acme Corporation is headquartered in Chennai, India.",
}

CONVERSIONS = {("m", "ft"): lambda v: v * 3.28084, ("ft", "m"): lambda v: v / 3.28084,
               ("kg", "lb"): lambda v: v * 2.20462, ("lb", "kg"): lambda v: v / 2.20462,
               ("c", "f"): lambda v: v * 9 / 5 + 32, ("f", "c"): lambda v: (v - 32) * 5 / 9}


def handle_call(name: str, args: dict) -> dict:
    if name == "unit_convert":
        fn = CONVERSIONS.get((args["from_unit"], args["to_unit"]))
        if not fn:
            return {"content": [{"type": "text", "text": "unsupported conversion"}], "isError": True}
        result = round(fn(float(args["value"])), 6)
        return {"content": [{"type": "text",
                             "text": f"{args['value']} {args['from_unit']} = {result} {args['to_unit']}"}],
                "isError": False}
    if name == "fake_web_lookup":
        key = str(args.get("key", "")).strip().lower()
        text = TABLE.get(key, f"No entry found for key '{key}'.")
        return {"content": [{"type": "text", "text": text}], "isError": False}
    if name == "slow_multiply":
        time.sleep(min(30.0, float(args.get("delay_s", 0))))
        return {"content": [{"type": "text", "text": str(float(args["a"]) * float(args["b"]))}],
                "isError": False}
    return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid, method = msg.get("id"), msg.get("method", "")
        if method == "initialize":
            result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "ara-demo-server", "version": "0.1.0"}}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            params = msg.get("params", {})
            result = handle_call(params.get("name", ""), params.get("arguments", {}) or {})
        elif method == "ping":
            result = {}
        else:
            if rid is not None:
                print(json.dumps({"jsonrpc": "2.0", "id": rid,
                                  "error": {"code": -32601, "message": f"unknown method {method}"}}),
                      flush=True)
            continue
        if rid is not None:
            print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}), flush=True)


if __name__ == "__main__":
    main()
