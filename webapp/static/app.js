(function () {
    "use strict";

    document.addEventListener("DOMContentLoaded", function () {
        const composer = document.getElementById("composer");
        const textarea = composer ? composer.querySelector("textarea") : null;
        const chat = document.getElementById("chat");
        const status = document.getElementById("pipeline-status");
        const modelSelect = document.getElementById("model-select");
        const tempRange = document.getElementById("temp-range");
        const tempVal = document.getElementById("temp-val");

        if (tempRange && tempVal) {
            tempRange.addEventListener("input", function () {
                tempVal.textContent = tempRange.value;
            });
        }

        document.querySelectorAll("[data-q]").forEach(function (btn) {
            btn.addEventListener("click", function () {
                if (!textarea) return;
                textarea.value = btn.getAttribute("data-q");
                textarea.focus();
                textarea.scrollIntoView({ behavior: "smooth", block: "center" });
            });
        });

        if (!composer || !textarea) return;

        function setStatus(text, level) {
            if (!status) return;
            status.classList.remove("hidden", "error");
            if (level === "error") status.classList.add("error");
            status.textContent = text;
        }

        function hideStatus() {
            if (!status) return;
            status.classList.add("hidden");
            status.classList.remove("error");
            status.textContent = "";
        }

        function setBusy(busy) {
            const send = composer.querySelector(".send");
            if (send) send.disabled = busy;
            composer.dataset.busy = busy ? "1" : "";
        }

        function appendUser(text) {
            if (!chat) return;
            const wrap = document.createElement("div");
            wrap.className = "turn user";
            const bubble = document.createElement("div");
            bubble.className = "bubble user-bubble";
            bubble.textContent = text;
            wrap.appendChild(bubble);
            chat.appendChild(wrap);
        }

        function parseSseChunk(chunk) {
            let event = "message";
            const dataLines = [];
            for (const line of chunk.split("\n")) {
                if (line.startsWith("event:")) event = line.slice(6).trim();
                else if (line.startsWith("data:")) dataLines.push(line.slice(5));
            }
            if (!dataLines.length) return { event, data: "" };
            return { event, data: dataLines.join("\n").trim() };
        }

        async function streamQuery(question) {
            const params = new URLSearchParams();
            params.set("question", question);
            if (modelSelect && modelSelect.value) params.set("model_id", modelSelect.value);
            if (tempRange) params.set("temperature", tempRange.value);

            const res = await fetch("/query", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: params.toString(),
            });

            if (!res.ok) {
                let detail = res.statusText;
                try {
                    const j = await res.json();
                    if (j.error) detail = j.error;
                } catch (_) {}
                throw new Error(detail);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const read = await reader.read();
                if (read.done) break;
                buffer += decoder.decode(read.value, { stream: true }).replace(/\r\n/g, "\n");

                let sep;
                while ((sep = buffer.indexOf("\n\n")) >= 0) {
                    const chunk = buffer.slice(0, sep);
                    buffer = buffer.slice(sep + 2);
                    if (!chunk.trim() || chunk.startsWith(":")) continue;
                    const { event, data } = parseSseChunk(chunk);
                    let payload = {};
                    try {
                        payload = JSON.parse(data);
                    } catch (_) {}

                    if (event === "stage") {
                        setStatus(payload.label || "Working...", payload.level);
                    } else if (event === "result" && payload.html) {
                        if (chat) chat.insertAdjacentHTML("beforeend", payload.html);
                        window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
                    } else if (event === "pipeline_error") {
                        setStatus(payload.message || "Pipeline error", "error");
                    }
                }
            }
        }

        composer.addEventListener("submit", async function (e) {
            e.preventDefault();
            const question = textarea.value.trim();
            if (!question || composer.dataset.busy === "1") return;

            textarea.value = "";
            appendUser(question);
            setStatus("Planning investigation...");
            setBusy(true);

            try {
                await streamQuery(question);
            } catch (err) {
                setStatus("Failed: " + err.message, "error");
            } finally {
                setBusy(false);
                setTimeout(hideStatus, 1800);
            }
        });
    });
})();
