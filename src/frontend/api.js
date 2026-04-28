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
        getSession: getSession,
        resetSession: resetSession,
    };
})(window);
