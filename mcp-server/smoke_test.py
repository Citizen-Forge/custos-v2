"""One-off manual smoke test: spawns server.py as a real MCP stdio
subprocess (same transport Claude Code would use) and drives it through
a real MCP client session against a real running api.py -- not a direct
Python function call, so this actually proves the MCP protocol wiring
works, not just that the underlying httpx calls do. Not a pytest --
prints results for a human to read, same posture as this project's
other probe_*.py scripts.

    CUSTOS_API_URL=http://host.docker.internal:8000 python smoke_test.py
"""

import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env={"CUSTOS_API_URL": os.environ.get("CUSTOS_API_URL", "http://localhost:8000")},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"tools registered ({len(names)}): {names}")
            expected = {
                "list_projects", "list_tickets", "get_ticket", "respond_to_ticket",
                "dismiss_ticket", "list_seats", "get_outcomes", "create_project",
                "create_epic", "create_story",
            }
            missing = expected - set(names)
            print(f"missing expected tools: {missing or 'none'}")

            result = await session.call_tool("list_projects", {})
            print(f"list_projects -> {result.content[0].text[:200]}")

            result = await session.call_tool(
                "create_project",
                {"name": "mcp-smoke-test project", "description": "created by smoke_test.py", "priority": 3},
            )
            print(f"create_project -> {result.content[0].text}")

            result = await session.call_tool("list_projects", {})
            found = "mcp-smoke-test project" in result.content[0].text
            print(f"new project visible in list_projects: {found}")

            result = await session.call_tool("get_ticket", {"issue_id": "does-not-exist-xyz"})
            print(f"get_ticket on unknown id -> {result.content[0].text[:150]}")


if __name__ == "__main__":
    asyncio.run(main())
