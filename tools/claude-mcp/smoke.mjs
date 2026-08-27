import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import path from "node:path";
import { fileURLToPath } from "node:url";

const directory = path.dirname(fileURLToPath(import.meta.url));
const child = spawn(process.execPath, [path.join(directory, "server.mjs")], {
  cwd: path.resolve(directory, "..", ".."),
  stdio: ["pipe", "pipe", "inherit"],
});
const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });

const timeout = setTimeout(() => {
  child.kill();
  process.stderr.write("실제 Claude MCP 왕복 시험이 150초 안에 끝나지 않았습니다.\n");
  process.exitCode = 1;
}, 150_000);

lines.on("line", (line) => {
  const message = JSON.parse(line);
  if (message.id === 1) {
    child.stdin.write(
      `${JSON.stringify({
        jsonrpc: "2.0",
        id: 2,
        method: "tools/call",
        params: {
          name: "ask_claude",
          arguments: {
            prompt: "Reply with exactly CLAUDE_MCP_OK and nothing else.",
            model: "haiku",
            effort: "low",
            timeout_seconds: 120,
          },
        },
      })}\n`,
    );
    return;
  }

  if (message.id === 2) {
    const answer = message.result?.content?.[0]?.text;
    if (message.result?.isError || answer !== "CLAUDE_MCP_OK") {
      process.stderr.write(`${JSON.stringify(message)}\n`);
      process.exitCode = 1;
    } else {
      process.stdout.write("CLAUDE_MCP_OK\n");
    }
    clearTimeout(timeout);
    child.stdin.end();
  }
});

child.on("close", (code) => {
  clearTimeout(timeout);
  if (code && process.exitCode === undefined) process.exitCode = code;
});

child.stdin.write(
  `${JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2025-06-18",
      capabilities: {},
      clientInfo: { name: "claude-mcp-smoke", version: "1.0.0" },
    },
  })}\n`,
);
