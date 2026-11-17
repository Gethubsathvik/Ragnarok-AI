// Incremental Server-Sent-Events parser.
// Feed it text chunks as they arrive; it calls onEvent(eventName, parsedJson) for each full message.
export function createSSEParser(onEvent) {
  let buf = "";
  return function feed(chunk) {
    buf += chunk.replace(/\r\n/g, "\n");
    let i;
    while ((i = buf.indexOf("\n\n")) !== -1) {
      const block = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let event = "message";
      const data = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length) onEvent(event, JSON.parse(data.join("\n")));
    }
  };
}
