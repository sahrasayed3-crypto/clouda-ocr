"use strict";

const state = { actionToken: null, lastPlanId: null, requestController: null };
const app = document.querySelector("#app");
const title = document.querySelector("#page-title");
const subtitle = document.querySelector("#page-subtitle");
const notice = document.querySelector("#notice");
const connection = document.querySelector("#connection-state");

const pages = {
  "overview": ["Overview", "Canonical local project state"],
  "datasets": ["Datasets", "Identity, lineage, safety, integrity, and loader compatibility"],
  "quality": ["Quality & Dedup", "Canonical quality gate and duplicate analysis"],
  "planner": ["Experiment Planner", "Deterministic plans over supported capabilities"],
  "preflight": ["Preflight", "Canonical readiness checks without training"],
  "models": ["Model Adapters", "Code, dependencies, assets, and validation"],
  "runs": ["Training Runs", "Experiment registry and checkpoint state"],
  "results": ["Results Store", "Persisted local evaluation evidence"],
  "benchmarks": ["Benchmarks", "Immutable local benchmark metadata"],
  "doctor": ["Doctor", "Normal and deep environment diagnostics"],
  "hardware": ["Hardware", "Doctor-derived local capability state"],
  "offline": ["Offline Status", "Fail-closed network and download policy"],
};

function node(tag, attributes = {}, children = []) {
  const element = document.createElement(tag);
  Object.entries(attributes).forEach(([key, value]) => {
    if (key === "class") element.className = value;
    else if (key === "text") element.textContent = String(value ?? "—");
    else if (key === "disabled") element.disabled = Boolean(value);
    else if (key.startsWith("on")) element.addEventListener(key.slice(2), value);
    else element.setAttribute(key, String(value));
  });
  const list = Array.isArray(children) ? children : [children];
  list.filter((child) => child !== null && child !== undefined).forEach((child) => {
    element.append(child instanceof Node ? child : document.createTextNode(String(child)));
  });
  return element;
}

function clear(element = app) { while (element.firstChild) element.removeChild(element.firstChild); }
function showNotice(message) { notice.textContent = message; notice.classList.add("visible"); }
function hideNotice() { notice.textContent = ""; notice.classList.remove("visible"); }
function loading() { clear(); app.append(node("div", { class: "state", text: "Loading canonical repository state…", "data-runtime-loading": "true" })); app.setAttribute("aria-busy", "true"); }
function empty(message = "No local records are available.") { clear(); app.append(node("div", { class: "state", text: message })); }
function errorView(error) { clear(); app.append(node("div", { class: "state danger-text", text: error.message || String(error) })); }
function pretty(value) { return JSON.stringify(value, null, 2); }
function statusClass(value) { return "status-" + String(value || "info").toLowerCase().replaceAll(" ", "-").replaceAll("_", "-"); }
function badge(value) { return node("span", { class: `badge ${statusClass(value)}`, text: value || "NOT AVAILABLE" }); }
function button(label, handler, options = {}) { const invoke = (event) => { try { const outcome = handler(event); if (outcome?.catch) outcome.catch(() => {}); } catch (failure) { showNotice(failure.message || String(failure)); } }; return node("button", { type: "button", class: options.class || "", disabled: options.disabled, title: options.title || "", onclick: invoke, text: label }); }
function details(label, value) { return node("details", {}, [node("summary", { text: label }), node("pre", { text: pretty(value) })]); }
function panel(label, children = []) { return node("section", { class: "panel" }, [node("h3", { text: label }), ...children]); }
function card(label, value, hint = "") { return node("article", { class: "card" }, [node("div", { class: "label", text: label }), value instanceof Node ? value : node("div", { class: "value", text: value }), node("div", { class: "hint", text: hint })]); }
function kv(data) { const list = node("dl", { class: "kv" }); Object.entries(data).forEach(([key, value]) => list.append(node("dt", { text: key }), node("dd", {}, value instanceof Node ? value : String(value ?? "NOT AVAILABLE")))); return list; }
function lineage(items) { const flow = node("div", { class: "lineage", "aria-label": "Lineage" }); items.forEach((item, index) => { if (index) flow.append(node("span", { class: "lineage-arrow", text: "→" })); flow.append(node("span", { class: "lineage-step", text: item })); }); return flow; }
function table(headers, rows) { const head = node("tr"); headers.forEach((header) => head.append(node("th", { text: header }))); const body = node("tbody"); rows.forEach((row) => { const tr = node("tr"); row.forEach((cell) => tr.append(node("td", {}, cell instanceof Node ? cell : String(cell ?? "—")))); body.append(tr); }); return node("div", { class: "table-wrap" }, node("table", {}, [node("thead", {}, head), body])); }
function pageHead(name, description, tools = []) { return node("div", { class: "page-head" }, [node("div", {}, [node("h2", { text: name }), node("p", { text: description })]), node("div", { class: "toolbar" }, tools)]); }

async function api(path, options = {}) {
  const init = { method: options.method || "GET", headers: { "Accept": "application/json" }, signal: state.requestController?.signal };
  if (options.body !== undefined) { init.headers["Content-Type"] = "application/json"; init.headers["X-Clouda-Lab-Action"] = state.actionToken || ""; init.body = JSON.stringify(options.body); }
  const response = await fetch(`/api/lab${path}`, init);
  const payload = await response.json().catch(() => ({ detail: "Invalid local response" }));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}
async function ensureSession() { if (!state.actionToken) state.actionToken = (await api("/session")).action_token; }
async function action(operation, message) { try { showNotice(message); await ensureSession(); const result = await operation(); showNotice("Canonical operation completed."); return result; } catch (failure) { showNotice(failure.message || String(failure)); throw failure; } }

async function renderOverview() {
  const data = await api("/overview"); const runs = data.runs || {};
  app.append(pageHead("Project status", "Every value is derived from local registries."));
  app.append(node("div", { class: "grid cards" }, [card("Datasets", data.datasets), card("Derived datasets", data.derived_datasets), card("Experiments", data.experiments), card("Planned / created", runs.CREATED || 0), card("Running", runs.RUNNING || 0), card("Interrupted", runs.INTERRUPTED || 0), card("Failed", runs.FAILED || 0), card("Completed", runs.COMPLETED || 0), card("Adapters available", data.available_model_adapters)]));
  app.append(node("div", { class: "grid two-col" }, [panel("Readiness", [kv({ "Repository": badge(data.repository), "GPU": badge(data.gpu), "Training stack": badge(data.training_readiness), "Doctor": badge(data.doctor), "Offline mode": badge(data.offline) })]), panel("Latest benchmark", [kv({ "Manifest": data.benchmark?.manifest?.benchmark_id, "Recorded results": data.benchmark?.results })])]));
}

async function renderDatasets(parts) {
  if (parts[1]) return renderDatasetDetail(parts[1]);
  const [datasetData, sourceData] = await Promise.all([api("/datasets"), api("/dataset-sources")]); const datasets = datasetData.datasets || []; const sources = sourceData.sources || []; const search = node("input", { type: "search", placeholder: "Search datasets", "aria-label": "Search datasets" }); const host = node("div");
  const draw = () => { clear(host); if (!datasets.length) return host.append(node("div", { class: "state", text: "No datasets found." })); const term = search.value.toLowerCase(); const visible = datasets.filter((item) => pretty(item).toLowerCase().includes(term)); if (!visible.length) return host.append(node("div", { class: "state", text: "No datasets match this search." })); host.append(table(["Dataset", "Version", "Rows", "Quality", "Safety", "Loader", "Updated"], visible.map((item) => [node("a", { href: `#/datasets/${encodeURIComponent(item.dataset_id)}`, text: item.dataset_id }), item.version, item.row_count, badge(item.quality?.status), badge(item.safety?.training_allowed ? "READY" : "BLOCKED"), badge(item.loader?.compatible ? "PASS" : "FAIL"), item.updated_at]))); };
  search.addEventListener("input", draw); app.append(pageHead("Local datasets", "Configured manifests and Results Store metadata are listed without scanning arbitrary roots.", [search]), host); draw();
  app.append(panel("Canonical source registry", [sources.length ? table(["Source", "Classification", "License", "Verified", "Download"], sources.map((source) => [source.name || source.source_id, badge(source.classification), source.license, source.license_verified ? "YES" : "NO", badge(source.download_enabled ? "AVAILABLE" : "DISABLED")])) : node("div", { class: "state", text: "No canonical source registry records are available." }), node("p", { class: "muted", text: "Source URLs and download controls are intentionally not exposed by Clouda Lab." })]));
}
async function renderDatasetDetail(datasetId) {
  const encoded = encodeURIComponent(datasetId); const [data, preview] = await Promise.all([api(`/datasets/${encoded}`), api(`/datasets/${encoded}/preview?limit=5`)]); const safety = data.safety || {};
  app.append(pageHead(data.dataset_id, data.identity, [node("a", { href: "#/datasets", text: "Back to datasets" })]));
  app.append(node("div", { class: `callout ${safety.training_allowed ? "" : "fail"}` }, [node("strong", { text: `Protected Holdout: ${safety.protected_holdout ? "YES" : "NO"}` }), node("span", { text: ` · Evaluation Only: ${safety.evaluation_only ? "YES" : "NO"} · Training Allowed: ${safety.training_allowed ? "YES" : "NO"}` })]));
  app.append(node("div", { class: "grid two-col" }, [panel("Identity", [kv({ "Canonical ID": data.dataset_id, "Version": data.version, "Manifest hash": data.manifest_hash, "Artifact root": data.artifact_root, "Rows / samples": data.row_count, "Pages": data.page_count })]), panel("Integrity & loader", [kv({ "Manifest readable": badge(data.integrity?.manifest_readable ? "PASS" : "FAIL"), "Row count": badge(data.integrity?.row_count_matches ? "PASS" : "FAIL"), "Loader compatibility": badge(data.loader?.compatible ? "PASS" : "FAIL"), "Reason": data.loader?.reason })])]));
  app.append(panel("Lineage", [data.lineage?.length ? lineage(data.lineage) : node("div", { class: "state", text: "No lineage metadata is recorded." })]));
  app.append(panel("Local record preview", [preview.records?.length ? table(["Sample ID", "Split", "Source", "Record"], preview.records.map((row) => [row.sample_id, row.target_split || row.split, row.source_id, details("Inspect", row)])) : node("div", { class: "state", text: "No preview records available." })]));
}

async function renderQuality() {
  const datasets = ((await api("/datasets")).datasets || []).filter((item) => item.source_type === "canonical_manifest"); if (!datasets.length) return empty("No canonical local manifest is available for quality analysis.");
  const select = node("select", { "aria-label": "Dataset" }); datasets.forEach((item) => select.append(node("option", { value: item.dataset_id, text: item.identity }))); const output = node("input", { placeholder: "Derived dataset label", "aria-label": "Derived dataset label" }); const result = node("div");
  async function refresh() { const data = await api(`/quality?dataset_id=${encodeURIComponent(select.value)}`); clear(result); result.append(panel("Quality state", [kv({ "Status": badge(data.status), "Verdict": badge(data.verdict), "Rejected samples": data.rejected_samples, "Duplicate clusters": data.duplicate_clusters }), data.report ? details("Canonical report", data.report) : node("p", { class: "muted", text: "No quality report has been run for this dataset." })])); }
  async function run(path, body) { const operation = await action(() => api(path, { method: "POST", body }), "Running the canonical local quality operation…"); await refresh(); if (path.endsWith("/derive")) result.append(panel("Derived dataset", [kv({ "Source dataset": operation.dataset_id, "Artifact": operation.artifact, "Derived identity": operation.derived_dataset?.dataset_version, "Rows": operation.derived_dataset?.clean_row_count }), details("Canonical lineage", operation.derived_dataset)])); }
  app.append(pageHead("Dataset quality", "Heavy scans run only when explicitly requested."), panel("Operations", [node("div", { class: "toolbar" }, [select, button("Run Quality Check", () => run("/quality/check", { dataset_id: select.value, max_samples: 5000 })), button("Analyze Duplicates", () => run("/quality/duplicates", { dataset_id: select.value, max_samples: 5000 }), { class: "secondary" }), output, button("Create Derived Dataset", () => run("/quality/derive", { dataset_id: select.value, output_label: output.value }), { class: "secondary" })])]), result); select.addEventListener("change", refresh); await refresh();
}

async function renderPlanner() {
  const options = await api("/planner/options"); const datasets = (options.datasets || []).filter((item) => item.training_allowed); const models = options.models || []; if (!datasets.length || !models.length) return empty("Planning requires a loader-compatible training dataset and a registered adapter.");
  const fields = { experiment_name: node("input", { placeholder: "Optional experiment name" }), adapter_type: node("select"), dataset_id: node("select"), precision: node("select"), seed: node("input", { type: "number", value: "20260723" }), batch_size: node("input", { type: "number", value: "1" }), gradient_accumulation_steps: node("input", { type: "number", value: "1" }), epochs: node("input", { type: "number", value: "1" }), max_steps: node("input", { type: "number", value: "100" }), learning_rate: node("input", { type: "number", step: "0.00001", value: "0.00005" }), checkpoint_frequency: node("input", { type: "number", value: "50" }) };
  models.forEach((item) => fields.adapter_type.append(node("option", { value: item.adapter_id, text: `${item.adapter_id} · ${item.model_family}` }))); datasets.forEach((item) => fields.dataset_id.append(node("option", { value: item.dataset_id, text: item.identity })));
  function selectAdapter() { const selected = models.find((item) => item.adapter_id === fields.adapter_type.value); if (!selected) return; clear(fields.precision); selected.supported_precision.forEach((item) => fields.precision.append(node("option", { value: item, text: item }))); }
  fields.adapter_type.addEventListener("change", selectAdapter); selectAdapter();
  const form = node("div", { class: "form-grid" }); Object.entries(fields).forEach(([name, control]) => form.append(node("label", {}, [name.replaceAll("_", " "), control]))); const result = node("div");
  async function create() { const body = {}; Object.entries(fields).forEach(([name, control]) => { body[name] = control.type === "number" ? Number(control.value) : control.value; }); const data = await action(() => api("/plans", { method: "POST", body }), "Generating a deterministic canonical plan…"); state.lastPlanId = data.plan_id; clear(result); result.append(panel("Generated plan", [kv({ "Plan ID": data.plan_id, "Model ID": data.model_id, "Dataset ID": data.dataset_id, "Configuration ID": data.config_id, "Expected Runtime Backend": data.expected_runtime_backend, "Output Path": data.output_path, "Execution": badge(data.execution_status) }), details("Canonical configuration", data.config), details("Resource plan", data.plan), button("Open Preflight", () => { location.hash = "#/preflight"; }, { class: "secondary" })])); }
  app.append(pageHead("Plan an experiment", "The plan is inspection-only and deterministic."), panel("Supported configuration", [form, node("div", { class: "spacer" }), button("Generate Plan", create)]), result);
}

async function renderPreflight() {
  const plans = (await api("/plans")).plans || []; if (!plans.length) return empty("Create an experiment plan before running preflight."); const select = node("select"); plans.forEach((item) => select.append(node("option", { value: item.plan_id, text: `${item.plan_id} · ${item.dataset_id}` }))); if (state.lastPlanId) select.value = state.lastPlanId; const result = node("div");
  async function run() { const data = await action(() => api("/preflight", { method: "POST", body: { plan_id: select.value, write_probe: false } }), "Running canonical preflight without a write probe…"); clear(result); result.append(panel("Preflight result", [kv({ "Final status": badge(data.final_status), "Ready": data.ready ? "YES" : "NO", "Adapter identity": data.adapter_identity, "Dataset identity": data.dataset_identity }), ...(data.sections || []).map((section) => panel(section.name, [badge(section.status), table(["Check", "Status", "Detail", "Blocker"], (section.checks || []).map((check) => [check.name, badge(check.status), check.detail, check.blocker ? "YES" : "NO"]))])), details("Canonical report", data)])); }
  app.append(pageHead("Validate a plan", "Hardware capability remains separate from logical validation."), panel("Plan selection", [node("div", { class: "toolbar" }, [select, button("Run Preflight", run)])]), result);
}

async function renderModels() {
  const models = (await api("/models")).models || []; if (!models.length) return empty("No registered model adapters are available."); app.append(pageHead("Registered adapters", "No model is instantiated and no asset is downloaded."));
  models.forEach((item) => app.append(panel(item.adapter_id, [node("div", { class: "grid three-col" }, [card("Code integration", badge(item.code_integration)), card("Local assets", badge(item.asset_status), item.availability_reason), card("GPU validation", badge(item.gpu_validation)), card("Real training", badge(item.real_training)), card("CPU mock capability", badge(item.cpu_compatibility ? "DECLARED" : "NOT DECLARED")), card("Dependencies", badge(item.dependency_check?.status), item.dependency_check?.detail)]), details("Capabilities", item.capabilities), details("Required optional dependencies", item.dependencies)])));
}

async function renderRuns(parts) {
  if (parts[1]) return renderRunDetail(parts[1]); const runs = (await api("/runs")).runs || []; app.append(pageHead("Experiment registry", "Runtime state comes from canonical metadata.")); if (!runs.length) return app.append(node("div", { class: "state", text: "No training runs recorded." })); app.append(table(["Run", "Experiment", "Model", "Dataset", "Status", "Step", "Started"], runs.map((run) => [node("a", { href: `#/runs/${encodeURIComponent(run.run_id)}`, text: run.run_id }), run.experiment_name, run.model_id, run.dataset_id, badge(run.status), run.final_step || run.last_completed_step, run.start_timestamp])));
}
async function renderRunDetail(runId) {
  const encoded = encodeURIComponent(runId); const [detail, checkpointData] = await Promise.all([api(`/runs/${encoded}`), api(`/runs/${encoded}/checkpoints`)]); const run = detail.run || {}; app.append(pageHead(run.run_id, run.experiment_name, [node("a", { href: "#/runs", text: "Back to runs" })]));
  app.append(node("div", { class: "grid two-col" }, [panel("Identity", [kv({ "Run ID": run.run_id, "Plan / config": run.config_hash, "Model": run.model_id, "Dataset": `${run.dataset_id}@${run.dataset_version}`, "Status": badge(run.status) })]), panel("Progress", [kv({ "Current step": run.final_step || run.last_completed_step, "Dataset rows": run.dataset_rows, "Latest loss": detail.metrics?.at(-1)?.loss, "Runtime state": badge(run.status) })])]));
  const checkpoints = checkpointData.checkpoints || []; const runLineage = [`${run.dataset_id}@${run.dataset_version}`, run.config_hash, run.run_id, ...checkpoints.map((item) => item.checkpoint_id)].filter((item) => item && !String(item).includes("undefined"));
  app.append(panel("Lineage", [runLineage.length ? lineage(runLineage) : node("div", { class: "state", text: "No run lineage metadata is recorded." })]));
  app.append(panel("Checkpoints", [checkpoints.length ? table(["Checkpoint", "Step", "Created", "Integrity", "Resume state"], checkpoints.map((item) => [item.checkpoint_id, item.step, item.timestamp, badge(item.integrity), badge(item.resume_state || "VALIDATION REQUIRED")])) : node("div", { class: "state", text: "No checkpoints are recorded." })]));
  if (run.status === "FAILED") app.append(node("div", { class: "callout fail", text: `Failure: ${run.error_type || "Runtime failure"} — ${run.error_message || "No additional sanitized metadata"}` }));
  const resumeResult = node("div"); async function checkResume() { const data = await action(() => api(`/runs/${encoded}/resume-check`, { method: "POST", body: {} }), "Validating canonical checkpoint compatibility…"); clear(resumeResult); resumeResult.append(panel(data.compatible ? "Resume Compatibility Passed" : "Resume Blocked", [kv({ "Execution": badge(data.status), "Checkpoint": data.checkpoint?.checkpoint_id || data.checkpoint?.path, "Integrity": badge(data.integrity), "Configuration Match": badge(data.configuration_match), "Dataset Match": badge(data.dataset_match), "Model Match": badge(data.model_match), "Reason": data.reason || data.execution_reason })])); }
  app.append(panel("Resume", [button("Validate Resume", checkResume, { class: "secondary" }), button("Start Real Training", () => {}, { disabled: true, title: "Training execution is not exposed by this service" }), node("p", { class: "muted", text: "Training and resume execution are not exposed by this service; canonical compatibility validation remains available." }), resumeResult]));
}

async function renderResults(parts) {
  if (parts[1]) {
    const data = await api(`/results/${encodeURIComponent(parts[1])}?metric_limit=200`); const run = data.run || {};
    app.append(pageHead(run.run_id, "Canonical Results Store bundle", [node("a", { href: "#/results", text: "Back to results" })]));
    app.append(node("div", { class: "grid two-col" }, [panel("Run identity", [kv({ "Model": run.model_id, "Revision": run.model_revision, "Dataset": `${run.dataset_id}@${run.dataset_version}`, "Split": run.split, "Status": badge(run.status) })]), panel("Bundle integrity", [kv({ "Status": badge(data.integrity?.valid ? "PASS" : "FAIL"), "Pages": data.integrity?.pages, "Predictions": data.integrity?.predictions, "Metrics": data.integrity?.metrics, "Issues": data.integrity?.issues?.length || 0 })])]));
    app.append(panel("Summary", [details("Canonical run summary", data.summary)]));
    app.append(panel("Metrics", [data.metrics?.length ? table(["Metric", "Scope", "Page", "Value", "Normalization"], data.metrics.map((metric) => [metric.metric_name, metric.scope, metric.page_id, metric.value, metric.normalization])) : node("div", { class: "state", text: "No persisted metrics are available for this run." }), data.metrics_truncated ? node("p", { class: "muted", text: "Metric rows are limited to the first 200 records." }) : null]));
    return;
  }
  const filters = { model: node("input", { placeholder: "Model filter" }), dataset: node("input", { placeholder: "Dataset filter" }), experiment: node("input", { placeholder: "Experiment filter" }), benchmark: node("input", { placeholder: "Benchmark filter" }), date: node("input", { type: "date", "aria-label": "Date filter" }), status: node("select") };
  ["", "CREATED", "RUNNING", "COMPLETED", "FAILED", "INTERRUPTED"].forEach((item) => filters.status.append(node("option", { value: item, text: item || "All statuses" }))); const result = node("div");
  async function refresh() { const params = new URLSearchParams(); Object.entries(filters).forEach(([name, control]) => { if (control.value) params.set(name, control.value); }); const data = await api(`/results?${params}`); clear(result); result.append(node("div", { class: "grid cards" }, [card("Datasets", data.datasets?.length || 0), card("Models", data.models?.length || 0), card("Runs", data.runs?.length || 0)])); if (data.runs?.length) result.append(panel("Persisted runs", [table(["Run", "Model", "Dataset", "Status", "Created"], data.runs.map((run) => [node("a", { href: `#/results/${encodeURIComponent(run.run_id)}`, text: run.run_id }), run.model_id, run.dataset_id, badge(run.status), run.started_at]))])); else result.append(node("div", { class: "state", text: "No Results Store runs match these filters." })); }
  app.append(pageHead("Results Store", "Read-only access to persisted local evidence."), panel("Filters", [node("div", { class: "toolbar" }, [...Object.values(filters), button("Apply", refresh, { class: "secondary" })])]), result); await refresh();
}

async function renderBenchmarks() {
  const data = await api("/benchmarks"); app.append(pageHead("Local benchmark evidence", "Incomplete runs remain unranked."), panel("Manifest identity", [kv({ "Benchmark": data.manifest?.benchmark_id, "Manifest hash": data.manifest?.manifest_sha256, "Artifact hash": data.manifest?.artifact_sha256, "Source": data.source })])); if (!data.results?.length) return app.append(node("div", { class: "state", text: "No benchmark result is available for this filter." })); app.append(table(["Model", "Status", "CER", "WER", "Normalized CER", "Latency", "Hardware", "Rankable"], data.results.map((row) => [row.model, badge(row.status), row.cer, row.wer, row.normalized_arabic_cer, row.seconds_per_page, row.gpu, row.rankable ? "YES" : "NO"])));
}

async function renderDoctor() {
  const result = node("div"); async function show(data) { clear(result); if (data.status === "NOT RUN") return result.append(node("div", { class: "state", text: "Doctor has not been run in this service process." })); result.append(panel("Aggregate result", [kv({ "Status": badge(data.overall_status), "Readiness": data.readiness, "Mode": data.deep ? "Deep Check" : "Normal Check", "Timestamp": data.timestamp })])); (data.sections || []).forEach((section) => result.append(panel(section.name, [badge(section.status), table(["Check", "Status", "Required", "Message"], (section.checks || []).map((check) => [check.name, badge(check.status), check.required ? "YES" : "NO", check.message]))]))); }
  async function run(deep) { await show(await action(() => api("/doctor/run", { method: "POST", body: { deep } }), deep ? "Running the canonical deep Doctor check…" : "Running the canonical Doctor check…")); }
  app.append(pageHead("System Doctor", "Checks run only on explicit request."), panel("Diagnostics", [node("div", { class: "toolbar" }, [button("Normal Check", () => run(false)), button("Deep Check", () => run(true), { class: "secondary" })])]), result); await show(await api("/doctor/latest"));
}

async function renderHardware() {
  const data = await api("/hardware"); const gpuMessage = data.gpu?.available ? `Clouda Doctor detected ${data.gpu.devices?.length || 0} CUDA device(s).` : (data.gpu?.reason || "Clouda Doctor did not report an available CUDA device."); app.append(pageHead("Local hardware", "Capability discovery comes from Clouda Doctor."), node("div", { class: "grid two-col" }, [panel("CPU & platform", [kv({ "Status": badge(data.cpu?.status), "Machine": data.cpu?.machine, "Platform": data.cpu?.platform })]), panel("GPU / CUDA", [kv({ "Status": badge(data.gpu?.status), "CUDA available": data.gpu?.available ? "YES" : "NO", "CUDA version": data.gpu?.cuda_version, "BF16": data.gpu?.bf16_supported, "Devices": data.gpu?.devices?.length || 0, "Reason": data.gpu?.reason })])]), node("div", { class: `callout ${data.gpu?.available ? "" : "warn"}`, text: gpuMessage }), data.storage ? panel("Storage", [details("Canonical Doctor storage checks", data.storage)]) : null);
}

async function renderOffline() {
  const data = await api("/offline"); app.append(pageHead("Offline readiness", "Network-dependent behavior is fail-closed."), node("div", { class: "grid two-col" }, [panel("Policy", [kv({ "Network Required": data.network_required ? "Yes" : "No", "Automatic Model Download": data.automatic_model_download ? "Enabled" : "Disabled", "Automatic Dataset Download": data.automatic_dataset_download ? "Enabled" : "Disabled", "Remote Provider Calls": data.remote_provider_calls ? "Enabled" : "Disabled" })]), panel("Runtime", [kv({ "Offline Mode": badge(data.offline ? "ACTIVE" : "FAIL"), "Service Schema": data.schema_version, "Binding": data.binding })])]));
}

const renderers = { overview: renderOverview, datasets: renderDatasets, quality: renderQuality, planner: renderPlanner, preflight: renderPreflight, models: renderModels, runs: renderRuns, results: renderResults, benchmarks: renderBenchmarks, doctor: renderDoctor, hardware: renderHardware, offline: renderOffline };

async function route() {
  state.requestController?.abort(); state.requestController = new AbortController();
  hideNotice(); const parts = (location.hash.replace(/^#\/?/, "") || "overview").split("/").map(decodeURIComponent); const page = pages[parts[0]] ? parts[0] : "overview"; title.textContent = pages[page][0]; subtitle.textContent = pages[page][1]; document.querySelectorAll("nav a").forEach((link) => link.classList.toggle("active", link.dataset.page === page)); document.querySelector(".sidebar").classList.remove("open"); loading();
  try { await renderers[page](parts); app.querySelector("[data-runtime-loading]")?.remove(); app.setAttribute("aria-busy", "false"); connection.textContent = "Local service"; connection.className = "badge status-pass"; } catch (failure) { if (failure?.name === "AbortError") return; app.setAttribute("aria-busy", "false"); connection.textContent = "Service error"; connection.className = "badge status-fail"; errorView(failure); }
}

document.querySelector("#menu-toggle").addEventListener("click", () => document.querySelector(".sidebar").classList.toggle("open"));
window.addEventListener("hashchange", route);
route();
