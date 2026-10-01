/* Spot console — helpers, the state poll, the header.
 *
 * Nothing here holds authoritative state. The page renders what /api/state last said, so
 * two browsers and a reload mid-mission all agree.
 */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

async function post(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
    keepalive: true,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ── state ──────────────────────────────────────────────────────────────── */

const EMPTY_ROUTE = { pick: "", bin_teddy_bear: "", bin_bottle: "", bin_other: "", home: "", patrol: [] };
let state = { mission: null, teleop: false, stance: "sit", selection: { ...EMPTY_ROUTE }, groups: {} };
const stateListeners = [];

function onState(listener) { stateListeners.push(listener); }

function route() {
  return { ...EMPTY_ROUTE, ...(state.selection || {}) };
}

function pill(id, cls, value) {
  const node = $(id);
  if (!node) return;
  node.className = "pill" + (id === "#pill-mission" ? " wide" : "") + (cls ? " " + cls : "");
  $("em", node).textContent = value;
}

const POSTURE = { standing: "standing", sitting: "sitting", off: "sitting · off", unknown: "unknown" };

function renderHeader(next) {
  pill("#pill-link", next.halted ? "warn" : "ok", next.halted ? "stopped" : "connected");

  const battery = next.battery || {};
  if (battery.percentage === undefined) pill("#pill-battery", "", "—");
  else pill("#pill-battery", battery.percentage < 20 ? "warn" : "ok",
            `${battery.percentage}% ${battery.status === "CHARGING" ? "chg" : ""}`.trim());

  const lease = next.lease || {};
  if (lease.owner === undefined) pill("#pill-lease", "", "—");
  else if (lease.held) pill("#pill-lease", "ok", "console");
  else if (lease.owner === "nobody") pill("#pill-lease", "", "free");
  else pill("#pill-lease", "warn", lease.owner.split(":")[0]);

  const posture = next.posture && next.posture !== "unknown" ? next.posture : (next.stance === "stand" ? "standing" : "sitting");
  pill("#pill-stance", posture === "standing" ? "ok" : "", POSTURE[posture] || posture);
  pill("#pill-graph", next.graph ? "ok" : "", next.graph ? "localised" : "not localised");

  const vision = next.vision || {};
  if (!vision.ready) pill("#pill-vision", "warn", "offline");
  else pill("#pill-vision", next.detecting ? "live" : "ok", next.detecting ? `scanning · ${vision.backend}` : vision.backend);

  if (next.mission) pill("#pill-mission", "busy", `${next.mission_label} · ${next.elapsed}s`);
  else if (next.teleop) pill("#pill-mission", "live", "teleop");
  else pill("#pill-mission", "", next.halted ? "stopped — holding" : "idle");
}

async function pollState() {
  try {
    const next = await (await fetch("/api/state")).json();
    state = next;
    renderHeader(next);
    stateListeners.forEach((listener) => listener(next));
  } catch {
    pill("#pill-link", "warn", "no console");
  }
}
setInterval(pollState, 1000);
window.addEventListener("DOMContentLoaded", pollState);
