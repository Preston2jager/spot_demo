/* The display area.
 *
 * Camera views are MJPEG streams from /video/<view>; the map, the live scan and the 3D
 * reconstruction are WebGL; the log is the run log. AI is a layer, not a view: the server
 * draws the segmentation into Hand Camera and Body Cameras, and both also carry an SVG
 * layer with every object's outline, fetched from /api/vision/scene — what the operator
 * hovers and clicks to choose what Spot grasps.
 */
"use strict";

const VIEW_TITLES = {
  hand: "Hand Camera", cameras: "Body Cameras", rs_blend: "Depth Camera",
  map: "Waypoint Map", live: "Live Scan", recon: "3D Reconstruction", log: "Run Log",
};
const STREAMED = new Set(["hand", "cameras", "rs_blend"]);
const SCANNED = new Set(["hand", "cameras"]);
const WEBGL = new Set(["map", "live", "recon"]);
let currentView = "hand";
const viewListeners = [];
const mapListeners = [];

function onView(listener) { viewListeners.push(listener); }
function onMapLoaded(listener) { mapListeners.push(listener); }

$$(".tab").forEach((tab) => tab.addEventListener("click", () => showView(tab.dataset.view, true)));

onState((next) => {
  $$(".tab.rs").forEach((tab) => { tab.hidden = !next.realsense; });
  $$(".tab.live").forEach((tab) => { tab.hidden = !next.live_cloud; });
  $("#detect-toggle").dataset.on = next.detecting ? "1" : "0";
  $("#detect-state").textContent = next.detecting ? "on" : "off";
  document.documentElement.style.setProperty("--g-teddy_bear", (next.groups.teddy_bear || {}).color || "#ffaa3c");
  document.documentElement.style.setProperty("--g-bottle", (next.groups.bottle || {}).color || "#32beff");
  document.documentElement.style.setProperty("--g-other", (next.groups.other || {}).color || "#5adc8c");
  syncChoice(next);
});

function feedUrl(view) {
  const alpha = view === "rs_blend" ? `&alpha=${($("#blend-alpha").value / 100).toFixed(2)}` : "";
  return `/video/${view}?t=${Date.now()}${alpha}`;
}

/* `byOperator` is a tab click: while a grasp choice is waiting, switching between Hand and
 * Body Cameras also switches which scan the choice is made from. */
function showView(view, byOperator = false) {
  if (byOperator && SCANNED.has(view) && choiceWaiting() && view !== state.pending_view) {
    post("/api/sort/confirm", { source: view }).catch((error) => complain(error.message));
  }
  currentView = view;
  const streamed = STREAMED.has(view);
  const stage = $(".panel.video");

  $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === view));
  $("#view-title").textContent = VIEW_TITLES[view] || "Camera";
  $("#feed-wrap").hidden = !streamed;
  $("#map-stage").hidden = view !== "map";
  $("#live-stage").hidden = view !== "live";
  $("#recon-stage").hidden = view !== "recon";
  $("#log-stage").hidden = view !== "log";
  stage.classList.toggle("map", WEBGL.has(view));
  stage.classList.toggle("plain", view === "log");
  stage.classList.toggle("rs", view === "rs_blend");
  // Detached rather than hidden: an attached MJPEG response keeps the robot capturing.
  if (streamed) $("#feed").src = feedUrl(view); else $("#feed").removeAttribute("src");

  $("#blend-bar").hidden = view !== "rs_blend";
  $("#depth-key").hidden = view !== "rs_blend";
  $("#depth-readout").hidden = view !== "rs_blend";
  $("#vision-overlay").hidden = !SCANNED.has(view);
  closeChooser();
  watchFreshness(view === "rs_blend");
  watchScene(SCANNED.has(view));
  if (view === "map") loadMap();
  if (view === "live") openLive(); else stopLive();
  if (view === "recon") openRecon(); else stopRecon();
  viewListeners.forEach((listener) => listener(view));
  try { history.replaceState(null, "", `#${view}`); } catch { /* file:// or sandboxed */ }
}

$("#blend-alpha").addEventListener("change", () => {
  if (currentView === "rs_blend") $("#feed").src = feedUrl("rs_blend");
});

/* ── toasts: refusals and failures, wherever the operator is looking ────── */

function complain(message, ok = false) {
  if (typeof addLine === "function") {
    addLine({ time: new Date().toTimeString().slice(0, 8), level: ok ? "SUCCESS" : "WARNING", text: message, kind: "log" });
  }
  const toast = el("div", ok ? "toast ok" : "toast", message);
  $("#toasts").append(toast);
  setTimeout(() => toast.classList.add("gone"), 4500);
  setTimeout(() => toast.remove(), 5000);
}

/* ── the depth camera: freshness and depth probing ──────────────────────── */

let freshnessTimer = null;

function watchFreshness(on) {
  clearInterval(freshnessTimer);
  freshnessTimer = null;
  $("#feed-stale").hidden = true;
  if (!on) return;
  const check = async () => {
    try {
      const health = await fetch("/api/realsense/health").then((r) => r.json());
      const stale = [health.color, health.depth].filter((s) => s && s.stale);
      $("#feed-stale").hidden = stale.length === 0;
      if (stale.length) $("#feed-stale").textContent = `frozen — last frame ${Math.round(Math.max(...stale.map((s) => s.age || 0)))}s ago`;
    } catch {
      $("#feed-stale").hidden = false;
      $("#feed-stale").textContent = "bridge unreachable";
    }
  };
  check();
  freshnessTimer = setInterval(check, 5000);
}

/* Where the picture actually sits inside its letterboxed <img>. */
function pictureRect(image) {
  const box = image.getBoundingClientRect();
  const { naturalWidth: nw, naturalHeight: nh } = image;
  if (!nw || !nh) return box;
  const scale = Math.min(box.width / nw, box.height / nh);
  return { left: box.left + (box.width - nw * scale) / 2, top: box.top + (box.height - nh * scale) / 2,
           width: nw * scale, height: nh * scale };
}

$("#feed").addEventListener("click", async (event) => {
  if (currentView !== "rs_blend") return;
  const picture = pictureRect($("#feed"));
  const u = (event.clientX - picture.left) / picture.width;
  const v = (event.clientY - picture.top) / picture.height;
  if (u < 0 || u > 1 || v < 0 || v > 1) return;
  $("#depth-note").textContent = "measuring...";
  try {
    const reading = await fetch(`/api/realsense/depth?u=${u.toFixed(4)}&v=${v.toFixed(4)}`).then((r) => r.json());
    if (reading.error) {
      $("#depth-value").textContent = "—";
      $("#depth-note").textContent = reading.error;
    } else if (reading.distance_m === null) {
      $("#depth-value").textContent = "no reading";
      $("#depth-note").textContent = `at ${reading.x}, ${reading.y} — nothing the sensor could measure`;
    } else {
      $("#depth-value").textContent = `${reading.distance_m.toFixed(2)} m`;
      $("#depth-note").textContent = `at ${reading.x}, ${reading.y} · ${reading.samples}/${reading.window} px · ${reading.age}s old`
        + (reading.aligned === false ? " · unaligned, approximate" : "");
    }
  } catch (error) {
    $("#depth-note").textContent = error.message;
  }
});

/* ── the AI layer on Hand and Body Cameras ──────────────────────────────────
 *
 * The picture is rendered on the server — grey scene, objects in colour, labels — and this
 * layer only adds what needs the mouse: hover and selection outlines, and the click. It is
 * drawn only while the server is drawing the same scan (`shown`), so the outlines never
 * sit over a live picture that has moved on.
 */

let scene = null;
let sceneTimer = null;
const SVG = "http://www.w3.org/2000/svg";

function watchScene(on) {
  clearInterval(sceneTimer);
  sceneTimer = null;
  scene = null;
  $("#vision-overlay").innerHTML = "";
  $("#choose-list").innerHTML = "";
  if (!on) return;
  fetchScene();
  sceneTimer = setInterval(fetchScene, 1000);
}

async function fetchScene() {
  const view = currentView;
  try {
    const res = await fetch(`/api/vision/scene?view=${view}`);
    if (view !== currentView) return;
    if (!res.ok) { $("#scene-note").textContent = state.detecting ? "scanning..." : "AI idle — no scan yet"; return; }
    const next = await res.json();
    const changed = !scene || next.seq !== scene.seq || next.selected !== scene.selected || next.shown !== scene.shown
      || JSON.stringify((next.tracks || []).map((t) => [t.id, t.hits]))
        !== JSON.stringify((scene.tracks || []).map((t) => [t.id, t.hits]));
    scene = next;
    const counts = Object.entries(next.counts).map(([g, n]) => `${n} ${g.replace("_", " ")}`).join(" · ");
    $("#scene-note").textContent = `scan ${next.seq} · ${next.age}s ago · ${Math.round(next.elapsed_ms)} ms · ${counts}`
      + ` · ${next.registered || 0} registered` + (next.held ? " · held for your choice" : "");
    if (changed) {
      drawOverlay();
      listChoices();
    }
  } catch { /* the next tick tries again */ }
}

function choiceWaiting() {
  return state.pending !== null && state.pending !== undefined;
}

function choosingHere() {
  return Boolean(scene) && choiceWaiting() && state.pending === scene.seq;
}

/* While choosing, the held scan's objects *are* the registered objects: each carries its
 * track number, one outline per camera that has it — "detected" in this scan, "remembered"
 * from an earlier one, or "projected" from its position (both drawn dashed). The picture
 * underneath is labelled with the same numbers. Otherwise it is this scan's detections. */
function drawOverlay() {
  const svg = $("#vision-overlay");
  svg.innerHTML = "";
  if (!scene || !scene.shown) return;
  svg.setAttribute("viewBox", `0 0 ${scene.width} ${scene.height}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
  const choosing = choosingHere() && Boolean(scene.tracks);
  svg.classList.toggle("choosing", choosing);
  const tracks = new Map((scene.tracks || []).map((t) => [t.id, t]));
  for (const obj of scene.objects) {
    if (obj.outline.length < 3) continue;
    const shape = document.createElementNS(SVG, "polygon");
    shape.setAttribute("points", obj.outline.map((p) => p.join(",")).join(" "));
    const title = document.createElementNS(SVG, "title");
    if (choosing && tracks.has(obj.id)) {
      const track = tracks.get(obj.id);
      shape.setAttribute("class", `outline track g-${track.uncertain ? "uncertain" : track.group}`
        + (obj.source === "detected" ? "" : " projected") + (obj.id === scene.selected ? " selected" : ""));
      shape.dataset.track = track.id;
      title.textContent = describeTrack(track) + (obj.source === "detected" ? "" : ` · outline ${obj.source}`);
      shape.addEventListener("click", (event) => openChooser(track, event));
      shape.addEventListener("mouseenter", () => hoverTrack(track.id, true));
      shape.addEventListener("mouseleave", () => hoverTrack(track.id, false));
    } else {
      shape.setAttribute("class", `outline g-${obj.uncertain ? "uncertain" : obj.group}`
        + (obj.id === scene.selected ? " selected" : ""));
      title.textContent = describe(obj);
    }
    shape.append(title);
    svg.append(shape);
  }
}

/* Every outline of one registered object lights up together. */
function hoverTrack(id, on) {
  $$(`#vision-overlay [data-track="${id}"]`).forEach((shape) => shape.classList.toggle("hot", on));
}

function describeTrack(track) {
  return `${track.id} · AI says '${track.label}'${track.uncertain ? " (uncertain)" : ""} · seen in ${track.hits} scans`
    + ` · ${Math.round(track.conf * 100)}% · sorts as ${track.group_label} · ${track.material} grip`;
}

function describe(obj) {
  return `#${obj.id} ${obj.label}${obj.uncertain ? " (uncertain)" : ""} ${Math.round(obj.conf * 100)}%`
    + ` · sorts as ${obj.group_label} · ${obj.material} grip`;
}

/* ── choosing what to grasp ─────────────────────────────────────────────── */

let announced = null;

function syncChoice(next) {
  const waiting = choiceWaiting();
  $("#choose-bar").hidden = !waiting;
  $$("[data-choose-view]").forEach((button) => {
    button.dataset.on = waiting && button.dataset.chooseView === next.pending_view ? "1" : "0";
  });
  const key = waiting ? `${next.pending}:${next.pending_view}` : null;
  if (waiting && announced !== key) {
    announced = key;
    if (currentView !== next.pending_view) showView(next.pending_view); else fetchScene();
  }
  if (!waiting) { announced = null; closeChooser(); }
  if (scene && $("#vision-overlay").classList.contains("choosing") !== choosingHere()) {
    drawOverlay();
    listChoices();
  }
}

function listChoices() {
  const list = $("#choose-list");
  list.innerHTML = "";
  if (!choosingHere()) return;
  const tracks = scene.tracks || [];
  if (!tracks.length) list.append(el("span", "muted", "nothing registered yet — Rescan to look again"));
  for (const track of tracks) {
    const seen = scene.objects.some((o) => o.id === track.id && o.source === "detected");
    const chip = el("button", `chip g-${track.uncertain ? "uncertain" : track.group}${seen ? "" : " projected"}`,
      `${track.id} ${track.label}${track.uncertain ? "?" : ""} ×${track.hits}`);
    chip.title = describeTrack(track);
    chip.addEventListener("click", (event) => openChooser(track, event));
    chip.addEventListener("mouseenter", () => hoverTrack(track.id, true));
    chip.addEventListener("mouseleave", () => hoverTrack(track.id, false));
    list.append(chip);
  }
}

$$("[data-choose-view]").forEach((button) => button.addEventListener("click", () => {
  showView(button.dataset.chooseView, true);
}));

$("#choose-rescan").addEventListener("click", async () => {
  try { await post("/api/sort/confirm", { rescan: true, source: state.pending_view || "cameras" }); }
  catch (error) { complain(error.message); }
});
$("#choose-cancel").addEventListener("click", closeChooser);

function openChooser(track, event) {
  if (!choosingHere()) return;
  const obj = track;
  const menu = $("#choose-menu");
  $("#choose-name").textContent = `${track.id} — ${track.label}`;
  $("#choose-meta").textContent = `seen in ${track.hits} scans · ${Math.round(track.conf * 100)}%`
    + `${track.uncertain ? " — uncertain" : ""} · sorts as ${track.group_label} · ${track.material} grip`;
  const bins = $("#choose-bins");
  bins.innerHTML = "";
  // Grasp-here missions set the object back down, so there is no bin to pick — one button.
  const options = (state.mission || "").startsWith("grab_")
    ? [{ group: obj.group, text: "Grasp this", cls: " predicted" }]
    : Object.entries(state.groups).map(([group, info]) => ({
      group,
      text: `Grasp → ${info.label} bin${group === obj.group ? " ✓" : ""}`,
      cls: group === obj.group ? " predicted" : "",
    }));
  const seq = scene.seq;
  for (const { group, text, cls } of options) {
    const button = el("button", `bin g-${group}${cls}`, text);
    button.addEventListener("click", async () => {
      closeChooser();
      try { await post("/api/sort/confirm", { scene: seq, track: track.id, group }); }
      catch (error) { complain(error.message); }
    });
    bins.append(button);
  }
  const stage = $(".video-stage").getBoundingClientRect();
  menu.hidden = false;
  const size = menu.getBoundingClientRect();
  menu.style.left = `${Math.max(8, Math.min(event.clientX - stage.left, stage.width - size.width - 8))}px`;
  menu.style.top = `${Math.max(8, Math.min(event.clientY - stage.top, stage.height - size.height - 8))}px`;
}

function closeChooser() {
  $("#choose-menu").hidden = true;
}

/* ── the map tab ────────────────────────────────────────────────────────── */

let map3d = null;
let cloudPoll = null;
let hoveredWaypoint = null;
let mapWaypoints = [];  // names, for the points window

function ensureMap() {
  if (map3d) return map3d;
  try {
    map3d = Map3D.mount($("#map3d"), { onPick: openWaypointMenu, onFrame: drawLabels });
  } catch (error) {
    $("#map-status").textContent = `3D view unavailable: ${error.message}`;
    return null;
  }
  if (!map3d) $("#map-status").textContent = "this browser has no WebGL — the 3D map cannot draw";
  return map3d;
}

/* The robot's map, or why there is none. Resolves to the waypoint names. */
async function fetchMap() {
  const res = await fetch("/api/map");
  const data = await res.json();
  if (!res.ok) {
    mapWaypoints = [];
    const error = new Error(data.error || "no map");
    error.local = data.local;
    error.empty = res.status === 404;
    throw error;
  }
  mapWaypoints = data.waypoints.map((w) => w.name).filter(Boolean).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  return data;
}

async function loadMap() {
  const view = ensureMap();
  if (!view) return;
  $("#map-empty").hidden = true;
  try {
    const data = await fetchMap();
    view.setGraph(data);
    view.setSelection(route());
    $("#map-source").textContent = `${data.waypoints.length} waypoints · ${data.source}`
      + (data.live ? " · localised" : " · on the robot, not localised");
    pollCloud();
  } catch (error) {
    view.setGraph({ waypoints: [], edges: [], fiducials: [] });
    view.clearCloud();
    $("#map-source").textContent = "";
    $("#map-status").textContent = error.empty ? "" : error.message;
    if (error.empty) {
      $("#map-empty").hidden = false;
      $("#map-load").hidden = !error.local;
      $("#map-load").textContent = error.local ? `Load ${error.local.split("/").pop()} onto Spot` : "";
    }
  }
  mapListeners.forEach((listener) => listener());
}

$("#map-load").addEventListener("click", async () => {
  try {
    await post("/api/map/load");
    $("#map-empty-note").textContent = "Uploading the map and localising — park Spot facing a fiducial...";
    pollState();
    const wait = setInterval(() => {
      if (state.mission === "load_map") return;
      clearInterval(wait);
      if (currentView === "map") loadMap();
    }, 1000);
  } catch (error) { complain(error.message); }
});

async function pollCloud() {
  clearInterval(cloudPoll);
  const tick = async () => {
    try {
      const status = await fetch("/api/map/cloud/status").then((r) => r.json());
      if (status.state === "building") {
        $("#map-status").textContent = `building the point cloud — ${status.done}/${status.total} waypoints`;
        return;
      }
      clearInterval(cloudPoll);
      cloudPoll = null;
      if (status.state === "error") { $("#map-status").textContent = status.error || "cloud failed"; return; }
      const points = new Float32Array(await fetch("/api/map/cloud").then((r) => r.arrayBuffer()));
      map3d.setCloud(points);
      $("#map-status").textContent = `${(points.length / 3).toLocaleString()} points`;
    } catch (error) {
      clearInterval(cloudPoll);
      cloudPoll = null;
      $("#map-status").textContent = error.message;
    }
  };
  await tick();
  if (!cloudPoll) cloudPoll = setInterval(tick, 1500);
}

const ROLE_LABELS = { pick: "pick", bin_teddy_bear: "teddy bin", bin_bottle: "bottle bin", bin_other: "other bin", home: "home" };

function drawLabels(positions) {
  const layer = $("#map-labels");
  const chosen = { ...EMPTY_ROUTE, ...(map3d ? map3d.selection : {}) };
  layer.innerHTML = "";
  for (const wp of positions) {
    const slots = Object.keys(ROLE_LABELS).filter((slot) => chosen[slot] === wp.name);
    const legs = chosen.patrol.map((name, i) => (name === wp.name ? i + 1 : 0)).filter(Boolean);
    if (!slots.length && !legs.length && wp.id !== hoveredWaypoint) continue;
    const tag = el("span", "map-label" + (slots.length ? ` ${slots[0]}` : legs.length ? " patrol" : ""));
    const roles = slots.map((slot) => ROLE_LABELS[slot]).concat(legs.length ? [`patrol ${legs.join("/")}`] : []);
    tag.textContent = roles.length ? `${wp.name} · ${roles.join(", ")}` : wp.name;
    tag.style.left = `${wp.screen.x}px`;
    tag.style.top = `${wp.screen.y}px`;
    layer.append(tag);
  }
}

function openWaypointMenu(hit, at) {
  const menu = $("#map-menu");
  if (!hit) { menu.hidden = true; hoveredWaypoint = null; map3d.draw(); return; }
  hoveredWaypoint = hit.id;
  const box = $(".video-stage").getBoundingClientRect();
  $("#map-menu-name").textContent = hit.name;
  menu.dataset.waypoint = hit.name;
  menu.hidden = false;
  menu.style.left = "0px";
  menu.style.top = "0px";
  const size = menu.getBoundingClientRect();
  menu.style.left = `${Math.max(0, Math.min(at.x - box.left, box.width - size.width - 8))}px`;
  menu.style.top = `${Math.max(0, Math.min(at.y - box.top, box.height - size.height - 8))}px`;
  map3d.draw();
}

$("#map-frame").addEventListener("click", () => { if (map3d) map3d.frame(); });

/* ── the live scan ──────────────────────────────────────────────────────── */

let live3d = null;
let livePoll = null;
let liveSeq = 0;
let livePoints = 0;

function ensureLive() {
  if (live3d) return live3d;
  try { live3d = Map3D.mount($("#live3d"), {}); } catch (error) {
    $("#live-status").textContent = `3D view unavailable: ${error.message}`;
    return null;
  }
  return live3d;
}

async function openLive() {
  const view = ensureLive();
  if (!view) return;
  try {
    const data = await fetchMap();
    view.setGraph(data);
    $("#live-source").textContent = `${data.waypoints.length} waypoints · ${data.source}`;
  } catch (error) { $("#live-source").textContent = error.message; }
  pollLive();
}

function stopLive() { clearInterval(livePoll); livePoll = null; }

function pollLive() {
  stopLive();
  const tick = async () => {
    let status;
    try {
      const res = await fetch("/api/map/live/status");
      status = await res.json();
      if (!res.ok) throw new Error(status.error || "the stitcher is not answering");
    } catch (error) { $("#live-status").textContent = error.message; return; }
    const unlocalised = status.state === "unlocalised";
    $("#live-localise").hidden = !unlocalised;
    if (unlocalised) $("#live-status").textContent = "not localised — park Spot facing a fiducial";
    else if (status.state === "error") $("#live-status").textContent = status.error || "the stitcher failed";
    else $("#live-status").textContent = `${livePoints.toLocaleString()} points · ${status.rate || 0} sweeps/s`;
    live3d.setRobot(status.robot || null);
    if (status.robot && $("#live-follow").dataset.on === "1") live3d.followRobot(status.robot);
    if (status.total && livePoints > status.total * 1.1) resetLive();
    if (status.seq > liveSeq) await pullLive();
  };
  tick();
  livePoll = setInterval(tick, 600);
}

async function pullLive() {
  try {
    const res = await fetch(`/api/map/live/cloud?since=${liveSeq}`);
    if (!res.ok) throw new Error((await res.json()).error || "no cloud");
    const seq = Number(res.headers.get("X-Cloud-Seq") || liveSeq);
    const points = new Float32Array(await res.arrayBuffer());
    live3d.appendCloud(points);
    livePoints += points.length / 3;
    liveSeq = seq;
  } catch (error) { $("#live-status").textContent = error.message; }
}

function resetLive() { liveSeq = 0; livePoints = 0; if (live3d) live3d.clearCloud(); }

$("#live-frame").addEventListener("click", () => { if (live3d) live3d.frame(); });
$("#live-follow").addEventListener("click", (event) => {
  event.currentTarget.dataset.on = event.currentTarget.dataset.on === "1" ? "0" : "1";
});

/* ── the 3D reconstruction ──────────────────────────────────────────────────
 *
 * Fused on the console's machine, not the robot's: this view polls its status (which also
 * keeps it running — it idles a few seconds after the last poll) and pulls the whole cloud
 * of weighted voxel centroids, in colour, whenever it has changed and at most once a second.
 * With AI on, the status also carries the registered objects — in the same vision frame as
 * the cloud — and each is labelled where it is, re-placed on every redraw as the view turns.
 */

let recon3d = null;
let reconPoll = null;
let reconSeq = -1;
let reconPulled = 0;
let reconFramed = false;
let reconObjects = [];

function ensureRecon() {
  if (recon3d) return recon3d;
  try { recon3d = Map3D.mount($("#recon3d"), { pointSize: 2.4, onFrame: drawReconLabels }); } catch (error) {
    $("#recon-status").textContent = `3D view unavailable: ${error.message}`;
    return null;
  }
  return recon3d;
}

function openRecon() {
  if (!ensureRecon()) return;
  stopRecon();
  const tick = async () => {
    let status;
    try { status = await fetch("/api/recon/status").then((r) => r.json()); }
    catch (error) { $("#recon-status").textContent = error.message; return; }
    for (const [name, info] of Object.entries(status.sources)) {
      const box = $(`#recon-sources [data-source="${name}"]`);
      if (box) box.checked = info.enabled;
      const rate = $(`#recon-sources [data-rate="${name}"]`);
      if (rate) {
        rate.textContent = info.note || (info.rate ? `${info.rate} fps` : "");
        rate.classList.toggle("warn", Boolean(info.note));
      }
    }
    reconObjects = status.objects || [];
    recon3d.setRobot(status.robot || null);
    if (status.robot && $("#recon-follow").dataset.on === "1") recon3d.followRobot(status.robot);
    $("#recon-source").textContent = `${(status.voxel_m * 100).toFixed(1)} cm voxels · ${status.voxels.toLocaleString()} fused`
      + (status.carved ? ` · ${status.carved.toLocaleString()} cleared` : "")
      + ` · RealSense mount ${status.rs_calibrated ? "calibrated" : "uncalibrated"}`;
    if (status.seq !== reconSeq && Date.now() - reconPulled > 1000) await pullRecon(status.seq);
  };
  tick();
  reconPoll = setInterval(tick, 700);
}

function stopRecon() { clearInterval(reconPoll); reconPoll = null; }

function drawReconLabels() {
  const layer = $("#recon-labels");
  layer.innerHTML = "";
  if (!recon3d) return;
  for (const obj of reconObjects) {
    const at = recon3d.toScreen([obj.x, obj.y, obj.z]);
    if (!at) continue;
    const tag = el("div", "recon-label" + (obj.uncertain ? " uncertain" : ""));
    tag.style.setProperty("--tag", obj.color);
    tag.style.left = `${at.x}px`;
    tag.style.top = `${at.y}px`;
    tag.append(el("i"), el("span", "", `${obj.id} ${obj.label}${obj.uncertain ? " ?" : ""} ${obj.conf.toFixed(2)}`));
    layer.append(tag);
  }
}

async function pullRecon(seq) {
  reconPulled = Date.now();
  try {
    const buffer = await fetch("/api/recon/cloud").then((r) => r.arrayBuffer());
    const count = new DataView(buffer).getUint32(0, true);
    const points = new Float32Array(buffer, 4, count * 3);
    const colours = new Uint8Array(buffer, 4 + count * 12, count * 3);
    recon3d.setColorCloud(points, colours);
    reconSeq = seq;
    $("#recon-status").textContent = `${count.toLocaleString()} points`;
    if (!reconFramed && count > 2000) { reconFramed = true; recon3d.frameCloud(); }
  } catch (error) { $("#recon-status").textContent = error.message; }
}

$$("#recon-sources input").forEach((box) => box.addEventListener("change", async () => {
  try { await post("/api/recon/sources", { [box.dataset.source]: box.checked }); }
  catch (error) { complain(error.message); }
}));
$("#recon-clear").addEventListener("click", async () => {
  try { await post("/api/recon/clear"); reconSeq = -1; reconFramed = false; if (recon3d) recon3d.clearCloud(); }
  catch (error) { complain(error.message); }
});
$("#recon-frame").addEventListener("click", () => { if (recon3d) recon3d.frameCloud(); });
$("#recon-calibrate").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = "Calibrating...";
  try {
    const result = await post("/api/recon/calibrate");
    reconSeq = -1;
    complain(`RealSense calibrated (${result.method}) — mount ${result.mount}, corrected ${result.turned_deg}° / ${result.moved_cm} cm`, true);
  } catch (error) { complain(`Calibration failed: ${error.message}`); }
  button.disabled = false;
  button.textContent = "Calibrate RealSense";
});
$("#recon-follow").addEventListener("click", (event) => {
  event.currentTarget.dataset.on = event.currentTarget.dataset.on === "1" ? "0" : "1";
});

/* A view named in the address bar opens straight onto it — #cameras, say. */
const initialView = location.hash.slice(1);
if (VIEW_TITLES[initialView] && initialView !== "hand") window.addEventListener("DOMContentLoaded", () => showView(initialView));
