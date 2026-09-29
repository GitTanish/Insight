(function () {
    "use strict";

    const STAGES = [
        { key: "planning", label: "Planning" },
        { key: "executing", label: "Executing" },
        { key: "validating", label: "Validating" },
        { key: "repairing", label: "Repairing", warn: true },
        { key: "rendering", label: "Rendering" },
        { key: "explaining", label: "Explaining" },
        { key: "grounding", label: "Verifying numbers", warn: true },
    ];

    const STAGE_BY_SERVER_KEY = {
        planning: "planning",
        planning_failed: "planning",
        executing: "executing",
        repairing: "repairing",
        rendering: "rendering",
        explaining: "explaining",
        grounding: "grounding",
    };

    document.addEventListener("DOMContentLoaded", function () {
        const composer = document.getElementById("composer");
        const textarea = composer ? composer.querySelector("textarea") : null;
        const chat = document.getElementById("chat");
        const status = document.getElementById("pipeline-status");
        const modelSelect = document.getElementById("model-select");
        const tempRange = document.getElementById("temp-range");
        const tempVal = document.getElementById("temp-val");
        const userKey = document.getElementById("user-key");
        const stopBtn = document.getElementById("stop-btn");

        if (userKey) {
            userKey.value = localStorage.getItem("insight_user_key") || "";
            userKey.addEventListener("change", function () {
                if (userKey.value.trim()) localStorage.setItem("insight_user_key", userKey.value.trim());
                else localStorage.removeItem("insight_user_key");
            });
        }

        if (tempRange && tempVal) {
            tempRange.addEventListener("input", function () {
                tempVal.textContent = tempRange.value;
            });
        }

        function apiKeyHeader() {
            const key = userKey && userKey.value.trim();
            return key ? { "X-Insight-Key": key } : {};
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

        let timer = null;
        let controller = null;

        function setError(text) {
            if (!status) return;
            status.classList.remove("hidden");
            status.classList.add("error");
            status.classList.remove("stages-on");
            status.textContent = text;
        }

        function setStatus(text) {
            if (!status) return;
            status.classList.remove("hidden", "error");
            status.classList.add("stages-on");
            status.textContent = text;
        }

        function resetStatus() {
            if (!status) return;
            status.classList.add("hidden");
            status.classList.remove("error", "stages-on");
            status.textContent = "";
        }

        function flash(message) {
            if (!status) return;
            status.classList.remove("hidden");
            status.classList.add("error");
            status.textContent = message;
        }

        function renderStages(activeKey) {
            if (!status) return;
            const items = STAGES.map(function (s) {
                let cls = "stage-step";
                if (s.key === activeKey) cls += s.warn ? " on warn" : " on";
                return "<span class=\"" + cls + "\">" + s.label + "</span>";
            });
            status.innerHTML =
                '<span class="stage-rail">' + items.join('<span class="stage-sep">›</span>') + "</span>" +
                '<span class="stage-label" id="stage-label"></span>' +
                '<span class="stage-timer" id="stage-timer">0.0s</span>';
        }

        function setStage(key, label) {
            renderStages(key);
            const el = document.getElementById("stage-label");
            if (el) el.textContent = label || "";
        }

        function startTimer() {
            const started = Date.now();
            if (timer) clearInterval(timer);
            timer = setInterval(function () {
                const el = document.getElementById("stage-timer");
                if (el) el.textContent = ((Date.now() - started) / 1000).toFixed(1) + "s";
            }, 100);
        }

        function stopTimer() {
            if (timer) clearInterval(timer);
            timer = null;
        }

        function setBusy(busy) {
            const send = composer.querySelector(".send");
            if (send) send.disabled = busy;
            if (stopBtn) stopBtn.hidden = !busy;
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
            wrap.scrollIntoView({ behavior: "smooth", block: "center" });
        }

        function bindFollowups(questions) {
            if (!chat || !questions || !questions.length) return;
            const wrap = document.createElement("div");
            wrap.className = "followups";
            const label = document.createElement("span");
            label.className = "mono-label";
            label.textContent = "Suggested next";
            wrap.appendChild(label);
            questions.forEach(function (q) {
                const btn = document.createElement("button");
                btn.className = "btn tiny followup";
                btn.textContent = q;
                btn.addEventListener("click", function () {
                    textarea.value = q;
                    textarea.focus();
                });
                wrap.appendChild(btn);
            });
            chat.appendChild(wrap);
        }

        let liveBubble = null;
        let liveText = "";
        let planPanel = null;

        function removePlanPanel() {
            if (planPanel && planPanel.parentNode) {
                planPanel.parentNode.removeChild(planPanel);
            }
            planPanel = null;
        }

        function showPlanReview(plan, question) {
            if (!chat) return;
            removePlanPanel();

            const wrap = document.createElement("div");
            wrap.className = "turn assistant";

            const bubble = document.createElement("div");
            bubble.className = "bubble assistant-bubble plan-review";

            const head = document.createElement("div");
            head.className = "plan-head";
            head.textContent = "Review the plan before it runs";
            bubble.appendChild(head);

            if (question) {
                const q = document.createElement("p");
                q.className = "plan-question";
                q.textContent = question;
                bubble.appendChild(q);
            }

            const editor = document.createElement("textarea");
            editor.className = "plan-editor";
            editor.rows = Math.min(18, (plan.steps || []).length * 3 + 6);
            editor.spellcheck = false;
            editor.setAttribute("aria-label", "Analysis plan JSON");
            editor.value = JSON.stringify(plan, null, 2);
            bubble.appendChild(editor);

            const summary = document.createElement("div");
            summary.className = "plan-steps";
            (plan.steps || []).forEach(function (step) {
                const chip = document.createElement("span");
                chip.className = "chip plan-step";
                chip.textContent = (step.step_id != null ? step.step_id + ": " : "") + step.operation;
                summary.appendChild(chip);
            });
            if (plan.objective) {
                const obj = document.createElement("p");
                obj.className = "plan-objective";
                obj.textContent = plan.objective;
                bubble.appendChild(obj);
            }
            bubble.appendChild(summary);

            const actions = document.createElement("div");
            actions.className = "plan-actions";

            const run = document.createElement("button");
            run.className = "btn primary";
            run.textContent = "Approve & run";
            run.addEventListener("click", async function () {
                let parsed;
                try {
                    parsed = JSON.parse(editor.value);
                } catch (err) {
                    flash("That plan is not valid JSON: " + err.message);
                    return;
                }
                removePlanPanel();
                resetStatus();
                setStage("executing", "Running your approved plan...");
                startTimer();
                setBusy(true);
                try {
                    await streamQuery(question, JSON.stringify(parsed));
                    stopTimer();
                    resetStatus();
                } catch (err) {
                    flash(String(err.message || err));
                    stopTimer();
                } finally {
                    setBusy(false);
                }
            });

            const edit = document.createElement("button");
            edit.className = "btn";
            edit.textContent = "Edit JSON";
            edit.addEventListener("click", function () {
                editor.classList.toggle("expanded");
                run.focus();
            });

            actions.appendChild(run);
            actions.appendChild(edit);
            bubble.appendChild(actions);

            const hint = document.createElement("p");
            hint.className = "plan-hint";
            hint.textContent =
                "Nothing has been executed yet. Edit the JSON if you want different steps, then approve.";
            bubble.appendChild(hint);

            wrap.appendChild(bubble);
            chat.appendChild(wrap);
            planPanel = wrap;
            window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
        }

        function ensureLiveBubble() {
            if (liveBubble) return liveBubble;
            if (!chat) return null;
            const wrap = document.createElement("div");
            wrap.className = "turn assistant";
            const bubble = document.createElement("div");
            bubble.className = "bubble assistant-bubble answer-md streaming";
            wrap.appendChild(bubble);
            chat.appendChild(wrap);
            liveBubble = wrap;
            liveText = "";
            return wrap;
        }

        function appendDelta(text) {
            const wrap = ensureLiveBubble();
            if (!wrap) return;
            const bubble = wrap.querySelector(".answer-md");
            liveText += text;
            bubble.textContent = liveText;
            if (chat) {
                const nearBottom =
                    window.innerHeight + window.scrollY >= document.body.offsetHeight - 220;
                if (nearBottom) {
                    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
                }
            }
        }

        function clearLiveBubble() {
            if (liveBubble && liveBubble.parentNode) {
                liveBubble.parentNode.removeChild(liveBubble);
            }
            liveBubble = null;
            liveText = "";
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

        async function streamQuery(question, approvedPlan) {
            const params = new URLSearchParams();
            params.set("question", question);
            if (modelSelect && modelSelect.value) params.set("model_id", modelSelect.value);
            if (tempRange) params.set("temperature", tempRange.value);
            if (approvedPlan) params.set("approved_plan", approvedPlan);

            controller = new AbortController();
            const res = await fetch("/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                    ...apiKeyHeader(),
                },
                body: params.toString(),
                signal: controller.signal,
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
                        setStage(
                            STAGE_BY_SERVER_KEY[payload.key] || payload.key,
                            payload.label
                        );
                    } else if (event === "validation") {
                        setStage("validating", payload.valid
                            ? "Validated " + payload.warning_count + " caveat(s)"
                            : "Validator rejected results");
                    } else if (event === "suggestions") {
                        bindFollowups(payload.followups || []);
                    } else if (event === "delta") {
                        appendDelta(payload.text || "");
                    } else if (event === "plan_review") {
                        showPlanReview(payload.plan, payload.question || "");
                        stopTimer();
                        setBusy(false);
                    } else if (event === "delta_reset") {
                        liveText = "";
                        const bubble = liveBubble && liveBubble.querySelector(".answer-md");
                        if (bubble) bubble.textContent = "";
                    } else if (event === "result" && payload.html) {
                        clearLiveBubble();
                        if (chat) chat.insertAdjacentHTML("beforeend", payload.html);
                        window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
                    } else if (event === "end") {
                        /* stream complete */
                    }
                }
            }
        }

        if (stopBtn) {
            stopBtn.addEventListener("click", function () {
                if (controller) controller.abort();
            });
        }

        composer.addEventListener("submit", async function (e) {
            e.preventDefault();
            const question = textarea.value.trim();
            if (!question || composer.dataset.busy === "1") return;

            textarea.value = "";
            appendUser(question);
            resetStatus();
            setStage("planning", "Planning investigation...");
            startTimer();
            setBusy(true);

            try {
                await streamQuery(question);
                stopTimer();
                resetStatus();
            } catch (err) {
                stopTimer();
                if (err.name === "AbortError") {
                    resetStatus();
                } else {
                    setError("Failed: " + err.message);
                }
            } finally {
                setBusy(false);
            }
        });
    });
})();
