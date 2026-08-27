let prompt = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  prompt += chunk;
});
process.stdin.on("end", () => {
  const reviewMode = process.argv.includes("Read,Glob,Grep");
  process.stdout.write(
    `${JSON.stringify({
      type: "result",
      subtype: "success",
      is_error: false,
      result: `${reviewMode ? "REVIEW" : "ASK"}:${prompt}`,
      session_id: "00000000-0000-4000-8000-000000000000",
      model: "mock-claude",
      duration_ms: 1,
      total_cost_usd: 0,
      num_turns: 1,
    })}\n`,
  );
});
