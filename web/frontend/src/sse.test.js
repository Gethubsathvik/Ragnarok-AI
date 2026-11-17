import test from "node:test";
import assert from "node:assert/strict";
import { createSSEParser } from "./sse.js";

const collect = () => {
  const events = [];
  return { events, feed: createSSEParser((e, d) => events.push([e, d])) };
};

test("parses several events in one chunk", () => {
  const { events, feed } = collect();
  feed('event: start\ndata: {"step":1}\n\nevent: token\ndata: {"text":"Hi"}\n\n');
  assert.deepEqual(events, [["start", { step: 1 }], ["token", { text: "Hi" }]]);
});

test("handles events split across arbitrary chunk boundaries", () => {
  const { events, feed } = collect();
  const raw = 'event: token\ndata: {"text":"héllo wörld"}\n\nevent: done\ndata: {"tokens":2}\n\n';
  for (let i = 0; i < raw.length; i += 3) feed(raw.slice(i, i + 3));
  assert.deepEqual(events, [["token", { text: "héllo wörld" }], ["done", { tokens: 2 }]]);
});

test("keeps newlines inside token text (JSON-escaped)", () => {
  const { events, feed } = collect();
  feed('event: token\ndata: {"text":"line1\\nline2\\n\\n"}\n\n');
  assert.equal(events[0][1].text, "line1\nline2\n\n");
});

test("ignores incomplete trailing message and CRLF endings", () => {
  const { events, feed } = collect();
  feed('event: token\r\ndata: {"text":"a"}\r\n\r\nevent: token\ndata: {"te');
  assert.equal(events.length, 1);
  feed('xt":"b"}\n\n');
  assert.deepEqual(events.map((e) => e[1].text), ["a", "b"]);
});
