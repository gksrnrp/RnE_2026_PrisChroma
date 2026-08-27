import { spawn } from "node:child_process";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import path from "node:path";

const SERVER_NAME = "rne-claude-code-bridge";
const SERVER_VERSION = "1.0.0";
const DEFAULT_PROTOCOL_VERSION = "2025-06-18";
const MAX_PROMPT_CHARS = 50_000;
const MAX_OUTPUT_CHARS = 2_000_000;
const DEFAULT_TIMEOUT_SECONDS = 180;
const MAX_TIMEOUT_SECONDS = 600;

const serverDirectory = path.dirname(fileURLToPath(import.meta.url));
const defaultProjectRoot = path.resolve(serverDirectory, "..", "..");
const projectRoot = path.resolve(
  process.env.CLAUDE_MCP_PROJECT_ROOT || defaultProjectRoot,
);

const tools = [
  {
    name: "ask_claude",
    title: "Ask Claude Code",
    description:
      "Claude Code CLI에 독립적인 의견을 묻습니다. 파일 도구는 비활성화되며 저장소를 수정하지 않습니다.",
    inputSchema: {
      type: "object",
      properties: {
        prompt: {
          type: "string",
          minLength: 1,
          maxLength: MAX_PROMPT_CHARS,
          description: "Claude에게 전달할 질문 또는 검토 요청",
        },
        model: {
          type: "string",
          enum: ["sonnet", "opus", "haiku", "fable"],
          description: "선택적 Claude 모델 별칭. 생략하면 Claude Code 기본값을 사용합니다.",
        },
        effort: {
          type: "string",
          enum: ["low", "medium", "high", "xhigh", "max"],
          description: "선택적 추론 노력 수준",
        },
        timeout_seconds: {
          type: "integer",
          minimum: 10,
          maximum: MAX_TIMEOUT_SECONDS,
          default: DEFAULT_TIMEOUT_SECONDS,
        },
      },
      required: ["prompt"],
      additionalProperties: false,
    },
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
  {
    name: "review_with_claude",
    title: "Review project with Claude Code",
    description:
      "Claude Code CLI가 현재 저장소를 읽기 전용으로 조사하고 독립 검토를 반환합니다. Read/Glob/Grep만 허용합니다.",
    inputSchema: {
      type: "object",
      properties: {
        prompt: {
          type: "string",
          minLength: 1,
          maxLength: MAX_PROMPT_CHARS,
          description: "검토할 질문, 범위, 기대 출력",
        },
        model: {
          type: "string",
          enum: ["sonnet", "opus", "haiku", "fable"],
          description: "선택적 Claude 모델 별칭. 생략하면 Claude Code 기본값을 사용합니다.",
        },
        effort: {
          type: "string",
          enum: ["low", "medium", "high", "xhigh", "max"],
          description: "선택적 추론 노력 수준",
        },
        timeout_seconds: {
          type: "integer",
          minimum: 10,
          maximum: MAX_TIMEOUT_SECONDS,
          default: DEFAULT_TIMEOUT_SECONDS,
        },
      },
      required: ["prompt"],
      additionalProperties: false,
    },
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
  },
];

function jsonRpcError(id, code, message, data) {
  const error = { code, message };
  if (data !== undefined) error.data = data;
  return { jsonrpc: "2.0", id: id ?? null, error };
}

function validateArguments(argumentsValue) {
  if (!argumentsValue || typeof argumentsValue !== "object") {
    throw new Error("arguments는 객체여야 합니다.");
  }

  const { prompt, model, effort } = argumentsValue;
  if (typeof prompt !== "string" || prompt.trim().length === 0) {
    throw new Error("prompt는 비어 있지 않은 문자열이어야 합니다.");
  }
  if (prompt.length > MAX_PROMPT_CHARS) {
    throw new Error(`prompt는 ${MAX_PROMPT_CHARS}자를 넘을 수 없습니다.`);
  }

  const allowedModels = new Set(["sonnet", "opus", "haiku", "fable"]);
  if (model !== undefined && !allowedModels.has(model)) {
    throw new Error("지원하지 않는 model입니다.");
  }

  const allowedEfforts = new Set(["low", "medium", "high", "xhigh", "max"]);
  if (effort !== undefined && !allowedEfforts.has(effort)) {
    throw new Error("지원하지 않는 effort입니다.");
  }

  const requestedTimeout = argumentsValue.timeout_seconds ?? DEFAULT_TIMEOUT_SECONDS;
  if (
    !Number.isInteger(requestedTimeout) ||
    requestedTimeout < 10 ||
    requestedTimeout > MAX_TIMEOUT_SECONDS
  ) {
    throw new Error(`timeout_seconds는 10~${MAX_TIMEOUT_SECONDS} 사이의 정수여야 합니다.`);
  }

  return {
    prompt,
    model,
    effort,
    timeoutMilliseconds: requestedTimeout * 1000,
  };
}

function claudeExecutable() {
  return process.env.CLAUDE_MCP_CLI || (process.platform === "win32" ? "claude.exe" : "claude");
}

function claudePrefixArguments() {
  const raw = process.env.CLAUDE_MCP_CLI_PREFIX_ARGS;
  if (!raw) return [];

  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed) || parsed.some((value) => typeof value !== "string")) {
    throw new Error("CLAUDE_MCP_CLI_PREFIX_ARGS는 문자열 배열 JSON이어야 합니다.");
  }
  return parsed;
}

function buildClaudeArguments(toolName, input) {
  const args = [
    ...claudePrefixArguments(),
    "-p",
    "--output-format",
    "json",
    "--no-session-persistence",
    "--permission-mode",
    "plan",
  ];

  if (toolName === "review_with_claude") {
    args.push(
      "--tools",
      "Read,Glob,Grep",
      "--allowed-tools",
      "Read,Glob,Grep",
      "--append-system-prompt",
      "읽기 전용 독립 검토입니다. 파일을 만들거나 수정하지 말고 명령을 실행하지 마세요. 근거가 되는 파일 경로를 밝혀 답하세요.",
    );
  } else {
    args.push(
      "--tools",
      "",
      "--append-system-prompt",
      "MCP를 통한 독립 자문입니다. 도구를 사용하거나 파일을 변경하지 말고 질문에 직접 답하세요.",
    );
  }

  if (input.model) args.push("--model", input.model);
  if (input.effort) args.push("--effort", input.effort);
  return args;
}

function extractClaudePayload(stdout) {
  const trimmed = stdout.trim();
  if (!trimmed) throw new Error("Claude Code CLI가 빈 출력을 반환했습니다.");

  try {
    return JSON.parse(trimmed);
  } catch {
    const lines = trimmed.split(/\r?\n/).reverse();
    for (const line of lines) {
      try {
        return JSON.parse(line);
      } catch {
        // Claude CLI가 JSON 앞에 진단 메시지를 쓴 경우 마지막 JSON 줄을 찾습니다.
      }
    }
    throw new Error("Claude Code CLI 출력을 JSON으로 해석하지 못했습니다.");
  }
}

function runClaude(toolName, input) {
  return new Promise((resolve, reject) => {
    let child;
    try {
      child = spawn(claudeExecutable(), buildClaudeArguments(toolName, input), {
        cwd: projectRoot,
        env: process.env,
        windowsHide: true,
        shell: false,
        stdio: ["pipe", "pipe", "pipe"],
      });
    } catch (error) {
      reject(error);
      return;
    }

    let stdout = "";
    let stderr = "";
    let settled = false;

    const finish = (callback) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      callback();
    };

    const timeout = setTimeout(() => {
      child.kill();
      finish(() => reject(new Error(`Claude Code CLI가 ${input.timeoutMilliseconds / 1000}초 안에 완료되지 않았습니다.`)));
    }, input.timeoutMilliseconds);

    child.on("error", (error) => finish(() => reject(error)));
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");

    child.stdout.on("data", (chunk) => {
      stdout += chunk;
      if (stdout.length > MAX_OUTPUT_CHARS) {
        child.kill();
        finish(() => reject(new Error("Claude Code CLI 출력이 허용 크기를 초과했습니다.")));
      }
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
      if (stderr.length > MAX_OUTPUT_CHARS) stderr = stderr.slice(-MAX_OUTPUT_CHARS);
    });

    child.on("close", (code) => {
      finish(() => {
        let payload;
        try {
          payload = extractClaudePayload(stdout);
        } catch (error) {
          const diagnostic = stderr.trim().slice(-4000);
          reject(new Error(diagnostic ? `${error.message}\n${diagnostic}` : error.message));
          return;
        }

        if (code !== 0 || payload.is_error) {
          const message = payload.result || payload.error || stderr.trim() || `종료 코드 ${code}`;
          reject(new Error(`Claude Code CLI 호출 실패: ${String(message).slice(0, 8000)}`));
          return;
        }

        resolve(payload);
      });
    });

    child.stdin.on("error", (error) => finish(() => reject(error)));
    child.stdin.end(input.prompt, "utf8");
  });
}

async function callTool(name, rawArguments) {
  if (!tools.some((tool) => tool.name === name)) {
    throw new Error(`알 수 없는 도구: ${name}`);
  }

  const input = validateArguments(rawArguments);
  const payload = await runClaude(name, input);
  const answer = typeof payload.result === "string" ? payload.result : JSON.stringify(payload.result);

  return {
    content: [{ type: "text", text: answer }],
    structuredContent: {
      session_id: payload.session_id ?? null,
      model: payload.model ?? null,
      duration_ms: payload.duration_ms ?? null,
      total_cost_usd: payload.total_cost_usd ?? null,
      num_turns: payload.num_turns ?? null,
    },
  };
}

async function handleRequest(request) {
  const id = request.id;
  switch (request.method) {
    case "initialize":
      return {
        jsonrpc: "2.0",
        id,
        result: {
          protocolVersion: request.params?.protocolVersion || DEFAULT_PROTOCOL_VERSION,
          capabilities: { tools: {} },
          serverInfo: { name: SERVER_NAME, version: SERVER_VERSION },
          instructions:
            "Claude Code CLI에 독립 의견을 요청하는 브리지입니다. 일반 자문은 ask_claude, 저장소 근거 검토는 review_with_claude를 사용하세요. 두 도구 모두 저장소를 수정하지 않습니다.",
        },
      };
    case "ping":
      return { jsonrpc: "2.0", id, result: {} };
    case "tools/list":
      return { jsonrpc: "2.0", id, result: { tools } };
    case "tools/call":
      try {
        return {
          jsonrpc: "2.0",
          id,
          result: await callTool(request.params?.name, request.params?.arguments),
        };
      } catch (error) {
        return {
          jsonrpc: "2.0",
          id,
          result: {
            content: [{ type: "text", text: error instanceof Error ? error.message : String(error) }],
            isError: true,
          },
        };
      }
    default:
      if (id === undefined) return null;
      return jsonRpcError(id, -32601, `지원하지 않는 메서드: ${request.method}`);
  }
}

function writeMessage(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
input.on("line", async (line) => {
  if (!line.trim()) return;

  let request;
  try {
    request = JSON.parse(line);
  } catch (error) {
    writeMessage(jsonRpcError(null, -32700, "JSON 파싱 오류", String(error)));
    return;
  }

  try {
    const response = await handleRequest(request);
    if (response) writeMessage(response);
  } catch (error) {
    writeMessage(jsonRpcError(request.id, -32603, "내부 서버 오류", String(error)));
  }
});

input.on("close", () => process.exit(0));
