import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import path from "node:path";
import { createInterface } from "node:readline";
import test from "node:test";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const serverPath = path.join(directory, "server.mjs");
const mockClaudePath = path.join(directory, "mock-claude.mjs");

function startServer() {
  const child = spawn(process.execPath, [serverPath], {
    cwd: path.resolve(directory, "..", ".."),
    env: {
      ...process.env,
      CLAUDE_MCP_CLI: process.execPath,
      CLAUDE_MCP_CLI_PREFIX_ARGS: JSON.stringify([mockClaudePath]),
    },
    stdio: ["pipe", "pipe", "pipe"],
  });
  const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
  const pending = [];
  lines.on("line", (line) => pending.shift()?.(JSON.parse(line)));

  return {
    child,
    request(message) {
      return new Promise((resolve) => {
        pending.push(resolve);
        child.stdin.write(`${JSON.stringify(message)}\n`);
      });
    },
  };
}

test("MCP 초기화, 도구 목록, Claude CLI 브리지를 처리한다", async (t) => {
  const server = startServer();
  t.after(async () => {
    server.child.stdin.end();
    await once(server.child, "close");
  });

  const initialized = await server.request({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2025-06-18",
      capabilities: {},
      clientInfo: { name: "test", version: "1" },
    },
  });
  assert.equal(initialized.result.serverInfo.name, "rne-claude-code-bridge");

  const listed = await server.request({ jsonrpc: "2.0", id: 2, method: "tools/list", params: {} });
  assert.deepEqual(
    listed.result.tools.map((tool) => tool.name),
    ["ask_claude", "review_with_claude"],
  );

  const asked = await server.request({
    jsonrpc: "2.0",
    id: 3,
    method: "tools/call",
    params: { name: "ask_claude", arguments: { prompt: "hello" } },
  });
  assert.equal(asked.result.content[0].text, "ASK:hello");
  assert.equal(asked.result.structuredContent.model, "mock-claude");

  const reviewed = await server.request({
    jsonrpc: "2.0",
    id: 4,
    method: "tools/call",
    params: { name: "review_with_claude", arguments: { prompt: "inspect" } },
  });
  assert.equal(reviewed.result.content[0].text, "REVIEW:inspect");

  const invalid = await server.request({
    jsonrpc: "2.0",
    id: 5,
    method: "tools/call",
    params: { name: "ask_claude", arguments: { prompt: "" } },
  });
  assert.equal(invalid.result.isError, true);
});
