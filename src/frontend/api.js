(function (global) {
    "use strict";

    async function parseErrorResponse(response) {
        try {
            const data = await response.json();
            if (data && typeof data.detail === "string") {
                return data.detail;
            }
            return JSON.stringify(data);
        } catch (_) {
            try {
                return await response.text();
            } catch (__)
            {
                return "Unknown API error";
            }
        }
    }

    async function requestJson(url, options) {
        const response = await fetch(url, options || {});
        if (!response.ok) {
            const detail = await parseErrorResponse(response);
            throw new Error(detail || ("HTTP " + response.status));
        }
        return response.json();
    }

    function getBootstrap() {
        return requestJson("/api/frontend/bootstrap");
    }

    function chat(payload) {
        return requestJson("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload || {}),
        });
    }

    async function chatStream(payload, onEvent) {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            body: JSON.stringify({ ...(payload || {}), stream: true }),
        });
        if (!response.ok) {
            const detail = await parseErrorResponse(response);
            throw new Error(detail || ("HTTP " + response.status));
        }
        if (!response.body) {
            throw new Error("Streaming not supported");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        function emitEvent(block) {
            const lines = block.split("\n");
            let eventName = "message";
            const dataLines = [];
            for (const line of lines) {
                if (line.startsWith("event:")) {
                    eventName = line.replace("event:", "").trim();
                } else if (line.startsWith("data:")) {
                    dataLines.push(line.replace("data:", "").trimStart());
                }
            }
            const dataStr = dataLines.join("\n");
            let data = dataStr;
            try {
                data = JSON.parse(dataStr);
            } catch (_) {
                // keep raw string
            }
            if (typeof onEvent === "function") {
                onEvent(eventName, data);
            }
        }

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parts = buffer.split("\n\n");
            buffer = parts.pop() || "";
            for (const part of parts) {
                if (part.trim()) {
                    emitEvent(part);
                }
            }
        }
        if (buffer.trim()) {
            emitEvent(buffer);
        }
    }

    function getSession(sessionId) {
        const sid = encodeURIComponent(String(sessionId || ""));
        return requestJson("/api/session/" + sid);
    }

    function resetSession(sessionId) {
        return requestJson("/api/session/reset", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId }),
        });
    }

    global.MragApi = {
        getBootstrap: getBootstrap,
        chat: chat,
        chatStream: chatStream,
        getSession: getSession,
        resetSession: resetSession,
    };
})(window);
