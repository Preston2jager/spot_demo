/* The control side: Stop, the AI switch, the command blocks, the points window, teleop,
 * and the run log. */
"use strict";

/* ── Stop ───────────────────────────────────────────────────────────────────
 *
 * Never disabled and never waits on anything else on the page. It is retried until the
 * console acknowledges it, because a Stop lost to one dropped wifi packet is the one
 * request that must not be lost.
 */

async function stopRobot() {
  $("#stop").classList.add("firing");
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      await post("/api/stop");
      break;
    } catch (error) {
      if (attempt === 4) complain(`Stop did not get through: ${error.message}`);
      await new Promise((resolve) => setTimeout(resolve, 150));
    }
  }
  held.clear();
  setTimeout(() => $("#stop").classList.remove("firing"), 400);
  pollState();
}

$("#stop").addEventListener("click", stopRobot);

$("#detect-toggle").addEventListener("click", async (event) => {
  const on = event.currentTarget.dataset.on === "1";
  try { await post(`/api/detect/${on ? "off" : "on"}`); pollState(); }
  catch (error) { complain(error.message); }
});

$("#surround-toggle").addEventListener("click", async (event) => {
  const on = event.currentTarget.dataset.on === "1";
  try { await post(`/api/vision/surround/${on ? "off" : "on"}`); pollState(); }
  catch (error) { complain(error.message); }
});

/* ── state → the command blocks ─────────────────────────────────────────── */

onState((next) => {
  const busy = Boolean(next.mission) || next.teleop;
  const posture = next.posture && next.posture !== "unknown" ? next.posture : (next.stance === "stand" ? "standing" : "sitting");
  const stance = posture === "standing" ? "stand" : "sit";
  $$(".cmd[data-mission]").forEach((button) => {
    const isTeleop = button.dataset.kind === "teleop";
    const running = button.dataset.mission === next.mission || (isTeleop && next.teleop);
    button.classList.toggle("running", running && !isTeleop);
    button.classList.toggle("active", isTeleop && next.teleop);
    button.classList.toggle("held", Boolean(button.dataset.stance) && button.dataset.stance === stance && !busy);
    button.disabled = busy && !running;
  });
  $("#teleop-hud").hidden = !next.teleop;
  $("#surround-toggle").dataset.on = next.surround ? "1" : "0";
  $("#surround-state").textContent = next.surround ? "all round" : "front only";
  $("#phase-line").textContent = next.mission
    ? (next.phase ? `▸ ${next.phase}` : "▸ running")
    : (next.halted ? "■ stopped — Spot is holding position" : "");
  $("#phase-line").classList.toggle("halted", !next.mission && next.halted);
  renderSetup();
});

$$(".cmd[data-mission]").forEach((button) => button.addEventListener("click", async () => {
  if (button.dataset.setup) { openSetup(button); return; }
  try {
    if (button.dataset.kind === "teleop") await post(`/api/teleop/${state.teleop ? "stop" : "start"}`);
    else await post(`/api/mission/${button.dataset.mission}`);
    pollState();
  } catch (error) { complain(error.message); }
}));

/* ── map points ─────────────────────────────────────────────────────────── */

async function setRoute(changes) {
  try {
    await post("/api/map/selection", { ...route(), ...changes });
    await pollState();
    if (map3d) map3d.setSelection(route());
  } catch (error) { complain(error.message); }
}

$$("#map-menu [data-slot]").forEach((button) => button.addEventListener("click", () => {
  $("#map-menu").hidden = true;
  setRoute({ [button.dataset.slot]: $("#map-menu").dataset.waypoint });
}));

$("#map-menu-patrol").addEventListener("click", () => {
  $("#map-menu").hidden = true;
  setRoute({ patrol: [...route().patrol, $("#map-menu").dataset.waypoint] });
});

$("#map-menu-go").addEventListener("click", async () => {
  $("#map-menu").hidden = true;
  try { await post("/api/navigate", { waypoint: $("#map-menu").dataset.waypoint }); pollState(); }
  catch (error) { complain(error.message); }
});

$("#live-clear").addEventListener("click", async () => {
  try { await post("/api/map/live/clear"); resetLive(); } catch (error) { complain(error.message); }
});

$("#live-localise").addEventListener("click", async () => {
  $("#live-status").textContent = "localising...";
  try { await post("/api/map/live/localise"); await post("/api/map/live/clear"); resetLive(); }
  catch (error) { $("#live-status").textContent = error.message; }
});

/* ── the points window ──────────────────────────────────────────────────────
 *
 * A mission that walks between waypoints opens this over the map instead of starting:
 * each point is a drop-down of the robot's waypoints (or a click on the map), and Start
 * dispatches only once every point is set and on the map Spot holds. It can be dragged by
 * its title, and it closes when the display switches away from the map.
 */

const SETUP = {
  sort: { slots: [["pick", "Pick point"], ["bin_teddy_bear", "Teddy bear bin"],
                  ["bin_bottle", "Bottle bin"], ["bin_other", "Other bin"]] },
  home: { slots: [["home", "Home point"]] },
  patrol: { patrol: true },
};
let setup = null;           // { mission, label, kind }
let setupDrawn = "";        // what the fields were last drawn from

function openSetup(button) {
  setup = { mission: button.dataset.mission, label: button.dataset.label, kind: button.dataset.setup };
  $("#setup-title").textContent = button.dataset.label;
  $("#setup-error").textContent = "";
  $("#setup").hidden = false;
  setupDrawn = "";
  if (currentView !== "map") showView("map"); else loadMap();
  renderSetup();
}

function closeSetup() {
  setup = null;
  $("#setup").hidden = true;
}

onView((view) => { if (view !== "map") closeSetup(); });
onMapLoaded(() => { setupDrawn = ""; renderSetup(); });

function waypointSelect(value, onChange, blank = "—") {
  const select = el("select");
  select.append(new Option(blank, ""));
  for (const name of mapWaypoints) select.append(new Option(name, name));
  if (value && !mapWaypoints.includes(value)) select.append(new Option(`${value} (not on Spot's map)`, value));
  select.value = value || "";
  select.addEventListener("change", () => onChange(select.value));
  return select;
}

function renderSetup() {
  if (!setup) return;
  const chosen = route();
  const key = JSON.stringify([setup.mission, chosen, mapWaypoints.length]);
  if (key === setupDrawn || $("#setup").contains(document.activeElement) && document.activeElement.tagName === "SELECT") return;
  setupDrawn = key;
  const fields = $("#setup-fields");
  fields.innerHTML = "";
  const spec = SETUP[setup.kind];
  $("#setup-note").textContent = !mapWaypoints.length ? "Spot has no map loaded — load one on the map first."
    : spec.patrol ? "Add stops in the order Spot should walk them, or click waypoints on the map."
      : "Choose each point below, or click a waypoint on the map.";

  for (const [slot, label] of spec.slots || []) {
    const row = el("label", `setup-row slot-${slot}`);
    row.append(el("span", "", label), waypointSelect(chosen[slot], (value) => setRoute({ [slot]: value })));
    fields.append(row);
  }
  if (spec.patrol) {
    const list = el("ol", "setup-patrol");
    chosen.patrol.forEach((name, index) => {
      const item = el("li", "", name);
      const remove = el("button", "ghost tiny", "✕");
      remove.title = "Remove this stop";
      remove.addEventListener("click", () => setRoute({ patrol: chosen.patrol.filter((_, i) => i !== index) }));
      item.append(remove);
      list.append(item);
    });
    if (!chosen.patrol.length) list.append(el("li", "muted", "no stops yet"));
    const add = el("div", "setup-row");
    add.append(waypointSelect("", (value) => { if (value) setRoute({ patrol: [...chosen.patrol, value] }); }, "add a stop…"));
    const clear = el("button", "ghost tiny", "clear");
    clear.addEventListener("click", () => setRoute({ patrol: [] }));
    add.append(clear);
    fields.append(list, add);
  }
}

function setupProblem() {
  const chosen = route();
  if (!mapWaypoints.length) return "Spot has no map loaded — load one on the map first.";
  const spec = SETUP[setup.kind];
  for (const [slot, label] of spec.slots || []) {
    if (!chosen[slot]) return `${label} is not set.`;
    if (!mapWaypoints.includes(chosen[slot])) return `${label} '${chosen[slot]}' is not on Spot's map.`;
  }
  if (spec.patrol) {
    if (chosen.patrol.length < 2) return "A patrol needs at least two stops.";
    const missing = chosen.patrol.find((name) => !mapWaypoints.includes(name));
    if (missing) return `Patrol stop '${missing}' is not on Spot's map.`;
  }
  return "";
}

$("#setup-start").addEventListener("click", async () => {
  if (!setup) return;
  const problem = setupProblem();
  $("#setup-error").textContent = problem;
  if (problem) return;
  try {
    await post(`/api/mission/${setup.mission}`);
    closeSetup();
    pollState();
  } catch (error) { $("#setup-error").textContent = error.message; }
});

$("#setup-close").addEventListener("click", closeSetup);

/* Dragged by its title bar, kept inside the display area. */
$("#setup-head").addEventListener("pointerdown", (event) => {
  if (event.target.closest("button")) return;
  const panel = $("#setup");
  const stage = $(".video-stage").getBoundingClientRect();
  const start = panel.getBoundingClientRect();
  const dx = event.clientX - start.left;
  const dy = event.clientY - start.top;
  const move = (e) => {
    const x = Math.max(0, Math.min(e.clientX - dx - stage.left, stage.width - start.width));
    const y = Math.max(0, Math.min(e.clientY - dy - stage.top, stage.height - start.height));
    panel.style.left = `${x}px`;
    panel.style.top = `${y}px`;
    panel.style.right = "auto";
  };
  const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
});

/* ── teleop ─────────────────────────────────────────────────────────────── */

const DRIVE = { w: { vx: 0.5 }, s: { vx: -0.5 }, a: { vy: 0.5 }, d: { vy: -0.5 }, q: { vrot: 0.8 }, e: { vrot: -0.8 } };
const held = new Set();
let driveTimer = null;

function driveVector() {
  const command = { vx: 0, vy: 0, vrot: 0 };
  held.forEach((key) => {
    command.vx += DRIVE[key].vx || 0;
    command.vy += DRIVE[key].vy || 0;
    command.vrot += DRIVE[key].vrot || 0;
  });
  return command;
}

/* Each command expires after ~0.6 s on the robot, so a browser that dies mid-stride stops Spot. */
function pump() {
  if (!state.teleop || held.size === 0) return;
  post("/api/teleop/drive", driveVector()).catch(() => {});
}

function releaseKeys() {
  held.clear();
  $$("kbd").forEach((key) => key.classList.remove("down"));
  clearInterval(driveTimer);
  driveTimer = null;
}

document.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea, select")) return;
  const key = event.key.toLowerCase();
  // Esc releases teleop if it is on, and otherwise is Stop — whatever else is happening.
  if (key === "escape") {
    event.preventDefault();
    if (state.teleop && !state.mission) { releaseKeys(); post("/api/teleop/stop").then(pollState).catch(() => {}); }
    else stopRobot();
    return;
  }
  if (!state.teleop || !DRIVE[key] || event.repeat) return;
  event.preventDefault();
  held.add(key);
  $(`kbd[data-key="${key}"]`)?.classList.add("down");
  pump();
  if (!driveTimer) driveTimer = setInterval(pump, 120);
});

document.addEventListener("keyup", (event) => {
  const key = event.key.toLowerCase();
  if (!DRIVE[key]) return;
  held.delete(key);
  $(`kbd[data-key="${key}"]`)?.classList.remove("down");
  if (held.size === 0) {
    releaseKeys();
    if (state.teleop) post("/api/teleop/drive", { vx: 0, vy: 0, vrot: 0 }).catch(() => {});
  }
});

window.addEventListener("blur", () => {
  if (held.size === 0) return;
  releaseKeys();
  if (state.teleop) post("/api/teleop/drive", { vx: 0, vy: 0, vrot: 0 }).catch(() => {});
});

/* ── the run log ────────────────────────────────────────────────────────── */

const logEl = $("#log");
let follow = true;
let filter = "all";
let unseen = 0;
// The backlog replayed on connect is history, not news: only count what arrives after it.
let countAfter = Date.now() + 1500;
const MAX_LINES = 900;

function addLine(entry) {
  const kind = entry.kind && entry.kind.startsWith("task") ? entry.kind : entry.level.toLowerCase();
  const line = el("div", `line ${kind}`);
  line.dataset.level = entry.level;
  line.dataset.kind = entry.kind || "log";
  const label = entry.kind === "task-start" ? "▸ RUN" : entry.kind === "task-done" ? "✓ DONE"
    : entry.kind === "task-error" ? "✗ FAIL" : entry.level;
  line.append(el("time", "", entry.time), el("span", "lvl", label), el("span", "msg", entry.text));
  applyFilter(line);
  logEl.append(line);
  while (logEl.childElementCount > MAX_LINES) logEl.firstElementChild.remove();
  if (follow && currentView === "log") logEl.scrollTop = logEl.scrollHeight;
  // Warnings arriving while the log is not on screen are counted on its tab.
  if (currentView !== "log" && Date.now() > countAfter && ["WARNING", "ERROR", "CRITICAL"].includes(entry.level)) {
    unseen += 1;
    $("#log-badge").textContent = unseen > 99 ? "99+" : String(unseen);
    $("#log-badge").hidden = false;
  }
}

onView((view) => {
  if (view !== "log") return;
  unseen = 0;
  $("#log-badge").hidden = true;
  if (follow) logEl.scrollTop = logEl.scrollHeight;
});

function applyFilter(line) {
  const isTask = line.dataset.kind.startsWith("task");
  const isWarn = ["WARNING", "ERROR", "CRITICAL"].includes(line.dataset.level);
  line.classList.toggle("hidden", !(filter === "all" || (filter === "task" && isTask) || (filter === "warn" && isWarn)));
}

$$(".filter").forEach((button) => button.addEventListener("click", () => {
  $$(".filter").forEach((other) => other.classList.toggle("active", other === button));
  filter = button.dataset.filter;
  $$(".line", logEl).forEach(applyFilter);
}));

$("#follow").addEventListener("click", (event) => {
  follow = event.currentTarget.dataset.on !== "1";
  event.currentTarget.dataset.on = follow ? "1" : "0";
  if (follow) logEl.scrollTop = logEl.scrollHeight;
});

logEl.addEventListener("scroll", () => {
  const atBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 40;
  if (!atBottom && follow) { follow = false; $("#follow").dataset.on = "0"; }
});

function connectLogs() {
  const source = new EventSource("/api/logs/stream");
  // Every connection replays the backlog, so a reconnect starts from a clean slate.
  source.onopen = () => { logEl.innerHTML = ""; };
  source.onmessage = (event) => addLine(JSON.parse(event.data));
  source.onerror = () => { source.close(); setTimeout(connectLogs, 2000); };
  countAfter = Date.now() + 1500;
}
connectLogs();
