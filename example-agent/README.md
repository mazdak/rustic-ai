# Example Agent: Doctor's Office Scheduler

This example is a **separate Rust project** that uses `rustic-ai` as a dependency.
It includes:

- A simple **MCP server** that exposes scheduling tools over JSON-RPC.
- A **Gemini-backed agent** that calls those MCP tools to schedule appointments.

## Prerequisites

- Rust toolchain (edition 2024)
- A Gemini API key in your environment:
  - `GEMINI_API_KEY` **or** `GOOGLE_API_KEY`

## Run the MCP server

From the `rustic-ai` repo root:

```bash
cargo run --manifest-path example-agent/Cargo.toml --bin mcp_server
```

This starts the server at `http://127.0.0.1:9099/rpc` by default.

## Run the agent

```bash
cargo run --manifest-path example-agent/Cargo.toml --bin assistant
```

By default, the agent uses:

- Model: `gemini:gemini-2.0-flash`
- MCP URL: `http://127.0.0.1:9099/rpc`
- An interactive prompt (press Enter to use the sample request)

You can override these:

```bash
cargo run --manifest-path example-agent/Cargo.toml --bin assistant \
  --model gemini:gemini-2.0-flash \
  --mcp-url http://127.0.0.1:9099/rpc \
  --prompt "I am patient P123. Book me with Dr. Smith on 2026-01-23 at 09:30 for a follow-up. My phone is 555-0100."
```

## Notes

- The MCP server is an in-memory demo. Restarting it clears appointments.
- The agent uses tool calling; it will ask for missing details instead of guessing.
