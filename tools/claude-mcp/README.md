# Claude Code MCP bridge

Codex가 Claude Code CLI에 독립 의견을 요청하도록 만든 로컬 STDIO MCP 서버다.
외부 npm 패키지 없이 Node.js 표준 라이브러리만 사용한다.

노출 도구:

- `ask_claude`: 파일 도구 없이 Claude에 질문한다.
- `review_with_claude`: 현재 저장소를 `Read`, `Glob`, `Grep`으로만 읽어 검토한다.

두 도구 모두 `claude -p --output-format json --permission-mode plan`을 사용하고
세션을 저장하지 않는다. MCP 표준 출력에는 JSON-RPC만 기록한다.

테스트:

```powershell
node --test tools/claude-mcp/server.test.mjs
```

실제 로그인된 Claude CLI까지 왕복 시험하려면 다음을 실행한다. 이 명령은 실제
Claude 사용량을 소비한다.

```powershell
node tools/claude-mcp/smoke.mjs
```

실제 Claude CLI 위치를 바꿔야 하면 `CLAUDE_MCP_CLI` 환경 변수를 사용한다.
기본값은 Windows에서 `claude.exe`, 그 외 환경에서 `claude`다.
