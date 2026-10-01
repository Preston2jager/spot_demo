/* The map, in three dimensions: the space Spot recorded, and the route through it.
 *
 * Hand-written WebGL rather than a library, for the same reason the rest of this console
 * has no dependencies: the laptop driving a demo is usually joined to Spot's own wifi
 * with no route to the internet, so a CDN script tag is a blank panel at the worst
 * possible moment. What is needed here is one buffer of points, one of lines, an orbit
 * camera and a click test — a few hundred lines, against three megabytes of scene graph
 * nobody would use the rest of.
 *
 * The points come from the waypoint snapshots of whatever map the *robot* is holding.
 * This robot has no lidar payload; these are what its depth cameras recorded while the
 * map was being walked, all put into the one seed frame the waypoints are anchored in.
 *
 * Two views are mounted from here, on two canvases. The Map tab draws that recorded
 * cloud in one go and lets the operator click a waypoint. The Live tab draws the same
 * graph while `appendCloud` adds what the robot's depth cameras are seeing *now*, sweep
 * by sweep, stitched into the same frame on the CoreIO's GPU — which is why the cloud
 * buffer grows rather than being replaced, and why there is a marker for the robot. The
 * 3D Recon tab uses `setColorCloud`: the laptop's fused reconstruction, in true colour.
 *
 * Coordinates: GraphNav's seed frame is z-up, which is what the camera below assumes.
 */
"use strict";

const Map3D = (() => {
  const VERTEX = `
    attribute vec3 aPosition;
    attribute vec3 aColor;
    uniform float uUseColor;  // 1: per-point colour from aColor (the reconstruction)
    uniform mat4 uMvp;
    uniform float uSize;
    uniform float uSolid;
    uniform vec3 uColor;
    uniform vec2 uHeight;     // low, high — the range the ramp spans
    varying vec3 vColor;

    // Height ramp: deep blue at the floor through cyan to a warm top. Colour carries the
    // one dimension a projected point cloud otherwise loses, which is what makes a ramp
    // read as a ramp and a table read as a table.
    vec3 ramp(float t) {
      vec3 low  = vec3(0.10, 0.16, 0.45);
      vec3 mid  = vec3(0.13, 0.83, 0.89);
      vec3 high = vec3(0.98, 0.78, 0.28);
      return t < 0.5 ? mix(low, mid, t * 2.0) : mix(mid, high, (t - 0.5) * 2.0);
    }

    void main() {
      gl_Position = uMvp * vec4(aPosition, 1.0);
      gl_PointSize = uSize;
      float t = clamp((aPosition.z - uHeight.x) / max(uHeight.y - uHeight.x, 0.001), 0.0, 1.0);
      vColor = uUseColor > 0.5 ? aColor : mix(ramp(t), uColor, uSolid);
    }`;

  const FRAGMENT = `
    precision mediump float;
    uniform float uRound;
    varying vec3 vColor;
    void main() {
      // Square points for the cloud (cheaper, and at one pixel nobody can tell), round
      // for the markers, where the shape is the difference between a dot and a speck.
      if (uRound > 0.5 && length(gl_PointCoord - vec2(0.5)) > 0.5) discard;
      gl_FragColor = vec4(vColor, 1.0);
    }`;

  /* ── small matrix library ───────────────────────────────────────────────── */

  function multiply(a, b) {
    const out = new Float32Array(16);
    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 4; col++) {
        out[col * 4 + row] =
          a[row] * b[col * 4] +
          a[4 + row] * b[col * 4 + 1] +
          a[8 + row] * b[col * 4 + 2] +
          a[12 + row] * b[col * 4 + 3];
      }
    }
    return out;
  }

  function perspective(fovY, aspect, near, far) {
    const f = 1 / Math.tan(fovY / 2);
    const out = new Float32Array(16);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far + near) / (near - far);
    out[11] = -1;
    out[14] = (2 * far * near) / (near - far);
    return out;
  }

  function lookAt(eye, target, up) {
    const z = normalise([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
    const x = normalise(cross(up, z));
    const y = cross(z, x);
    return new Float32Array([
      x[0], y[0], z[0], 0,
      x[1], y[1], z[1], 0,
      x[2], y[2], z[2], 0,
      -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
    ]);
  }

  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  function normalise(v) {
    const length = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / length, v[1] / length, v[2] / length];
  }

  function project(matrix, point, width, height) {
    const x = matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12];
    const y = matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13];
    const w = matrix[3] * point[0] + matrix[7] * point[1] + matrix[11] * point[2] + matrix[15];
    if (w <= 0) return null;  // behind the camera
    return { x: ((x / w) * 0.5 + 0.5) * width, y: (1 - ((y / w) * 0.5 + 0.5)) * height, w };
  }

  /* ── the view ───────────────────────────────────────────────────────────── */

  function mount(canvas, handlers) {
    const gl = canvas.getContext("webgl", { antialias: true, alpha: false });
    if (!gl) return null;

    const program = link(gl, VERTEX, FRAGMENT);
    const attribute = gl.getAttribLocation(program, "aPosition");
    const colorAttribute = gl.getAttribLocation(program, "aColor");
    const pointSize = handlers.pointSize || 1.6;
    const uniform = {
      mvp: gl.getUniformLocation(program, "uMvp"),
      size: gl.getUniformLocation(program, "uSize"),
      solid: gl.getUniformLocation(program, "uSolid"),
      color: gl.getUniformLocation(program, "uColor"),
      height: gl.getUniformLocation(program, "uHeight"),
      round: gl.getUniformLocation(program, "uRound"),
      useColor: gl.getUniformLocation(program, "uUseColor"),
    };

    const buffers = {
      cloud: gl.createBuffer(), edges: gl.createBuffer(),
      marks: gl.createBuffer(), fiducials: gl.createBuffer(),
      robot: gl.createBuffer(), heading: gl.createBuffer(), colors: gl.createBuffer(),
    };
    const counts = { cloud: 0, edges: 0, marks: 0, fiducials: 0, robot: 0, heading: 0 };

    // The cloud keeps a CPU-side mirror, because the live view appends to it: growing a
    // GL buffer means allocating a bigger one and re-uploading what was in the old, and
    // the alternative to holding a copy is reading half a million points back off the GPU.
    let cloudData = new Float32Array(0);
    let cloudCapacity = 0;
    let colored = false;
    let heightRange = [0, 2];
    let waypoints = [];          // {id, name, xyz}
    let selection = { pick: "", place: "", home: "", patrol: [] };
    let lastMvp = null;

    // Orbit camera. Azimuth and elevation in radians; distance in metres, because the
    // scene is in metres and a camera that thinks in anything else is a camera that
    // frames a different map wrongly.
    // Elevation starts high enough to read as a map — the thing an operator wants first
    // is "where are the waypoints", which is a plan view with enough tilt to keep the
    // third dimension visible. Orbiting down to eye level is one drag away.
    const camera = { azimuth: -Math.PI / 4, elevation: 0.95, distance: 18, target: [0, 0, 0] };

    function resize() {
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
      const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      gl.viewport(0, 0, canvas.width, canvas.height);
    }

    function matrix() {
      const aspect = canvas.width / Math.max(1, canvas.height);
      const cosE = Math.cos(camera.elevation);
      const eye = [
        camera.target[0] + camera.distance * cosE * Math.cos(camera.azimuth),
        camera.target[1] + camera.distance * cosE * Math.sin(camera.azimuth),
        camera.target[2] + camera.distance * Math.sin(camera.elevation),
      ];
      const far = Math.max(200, camera.distance * 12);
      return multiply(perspective(1.0, aspect, 0.1, far), lookAt(eye, camera.target, [0, 0, 1]));
    }

    function draw() {
      resize();
      gl.clearColor(0.02, 0.02, 0.03, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(program);

      const mvp = matrix();
      lastMvp = mvp;
      gl.uniformMatrix4fv(uniform.mvp, false, mvp);
      gl.uniform2fv(uniform.height, heightRange);
      gl.enableVertexAttribArray(attribute);

      if (counts.cloud) {
        bind(buffers.cloud);
        gl.uniform1f(uniform.size, pointSize);
        gl.uniform1f(uniform.solid, 0.0);
        gl.uniform1f(uniform.round, 0.0);
        if (colored) {
          gl.bindBuffer(gl.ARRAY_BUFFER, buffers.colors);
          gl.enableVertexAttribArray(colorAttribute);
          gl.vertexAttribPointer(colorAttribute, 3, gl.UNSIGNED_BYTE, true, 0, 0);
          gl.uniform1f(uniform.useColor, 1.0);
        }
        gl.drawArrays(gl.POINTS, 0, counts.cloud);
        gl.disableVertexAttribArray(colorAttribute);
        gl.uniform1f(uniform.useColor, 0.0);
      }
      if (counts.edges) {
        bind(buffers.edges);
        gl.uniform1f(uniform.solid, 1.0);
        gl.uniform3f(uniform.color, 0.35, 0.85, 0.92);
        gl.drawArrays(gl.LINES, 0, counts.edges);
      }
      // Fiducials before the waypoints, so a marker and a tag in the same place reads as
      // a waypoint at a fiducial rather than the other way round. White, because they are
      // the one thing in this scene that is neither the robot's route nor the room.
      if (counts.fiducials) {
        bind(buffers.fiducials);
        gl.uniform1f(uniform.size, 11.0);
        gl.uniform1f(uniform.solid, 1.0);
        gl.uniform1f(uniform.round, 0.0);
        gl.uniform3f(uniform.color, 0.93, 0.93, 0.95);
        gl.drawArrays(gl.POINTS, 0, counts.fiducials);
      }
      if (counts.marks) {
        bind(buffers.marks);
        gl.uniform1f(uniform.size, 9.0);
        gl.uniform1f(uniform.solid, 1.0);
        gl.uniform1f(uniform.round, 1.0);
        gl.uniform3f(uniform.color, 0.90, 0.11, 0.14);
        gl.drawArrays(gl.POINTS, 0, counts.marks);
      }
      if (counts.heading) {
        bind(buffers.heading);
        gl.uniform1f(uniform.solid, 1.0);
        gl.uniform3f(uniform.color, 0.20, 0.83, 0.60);
        gl.drawArrays(gl.LINES, 0, counts.heading);
      }
      if (counts.robot) {
        bind(buffers.robot);
        gl.uniform1f(uniform.size, 15.0);
        gl.uniform1f(uniform.solid, 1.0);
        gl.uniform1f(uniform.round, 1.0);
        gl.uniform3f(uniform.color, 0.20, 0.83, 0.60);
        gl.drawArrays(gl.POINTS, 0, counts.robot);
      }
      if (handlers.onFrame) handlers.onFrame(screenPositions());
    }

    function bind(buffer) {
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.vertexAttribPointer(attribute, 3, gl.FLOAT, false, 0, 0);
    }

    /* Where each waypoint currently is on screen — for the labels and the click test.
     * Done in JS rather than as a GPU read-back because there are dozens of waypoints,
     * not hundreds of thousands: the same projection, applied to the few things anybody
     * ever points at. */
    function screenPositions() {
      if (!lastMvp) return [];
      const ratio = window.devicePixelRatio || 1;
      const out = [];
      for (const waypoint of waypoints) {
        const point = project(lastMvp, waypoint.xyz, canvas.width / ratio, canvas.height / ratio);
        if (point) out.push({ ...waypoint, screen: point });
      }
      return out;
    }

    /* Where any point in the scene is on screen right now, or null behind the camera —
     * for labels pinned to things that are not waypoints. */
    function toScreen(xyz) {
      if (!lastMvp) return null;
      const ratio = window.devicePixelRatio || 1;
      return project(lastMvp, xyz, canvas.width / ratio, canvas.height / ratio);
    }

    /* ── data in ──────────────────────────────────────────────────────────── */

    /* Room for `floats` in the cloud buffer, keeping whatever is already in it. Doubling
     * rather than fitting exactly: a live view appends a few tens of thousands of points
     * twice a second, and reallocating on every sweep is the one thing that would make
     * this stutter. */
    function reserveCloud(floats) {
      if (floats <= cloudCapacity) return;
      const grown = Math.max(floats, cloudCapacity * 2, 1 << 18);
      const next = new Float32Array(grown);
      next.set(cloudData.subarray(0, counts.cloud * 3));
      cloudData = next;
      cloudCapacity = grown;
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.cloud);
      gl.bufferData(gl.ARRAY_BUFFER, cloudData.byteLength, gl.DYNAMIC_DRAW);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, cloudData.subarray(0, counts.cloud * 3));
    }

    function appendCloud(points) {
      colored = false;
      if (!points || !points.length) return;
      const start = counts.cloud * 3;
      reserveCloud(start + points.length);
      cloudData.set(points, start);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.cloud);
      gl.bufferSubData(gl.ARRAY_BUFFER, start * 4, points);
      counts.cloud += points.length / 3;
      rescaleHeights();
      draw();
    }

    function setCloud(points) {
      counts.cloud = 0;
      appendCloud(points);
    }

    function clearCloud() {
      counts.cloud = 0;
      draw();
    }

    /* Replace the cloud with points that carry their own colour — `colors` is RGB bytes,
     * one triplet per point. The reconstruction is re-sent whole, so this replaces. */
    function setColorCloud(points, colors) {
      counts.cloud = 0;
      appendCloud(points);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.colors);
      gl.bufferData(gl.ARRAY_BUFFER, colors, gl.DYNAMIC_DRAW);
      colored = true;
      draw();
    }

    /* Frame the cloud itself — for a view with no waypoints to frame. */
    function frameCloud() {
      const used = counts.cloud * 3;
      if (!used) return draw();
      const low = [Infinity, Infinity, Infinity];
      const high = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < used; i += 3 * 37) {
        for (let k = 0; k < 3; k++) {
          low[k] = Math.min(low[k], cloudData[i + k]);
          high[k] = Math.max(high[k], cloudData[i + k]);
        }
      }
      camera.target = [0, 1, 2].map((k) => (low[k] + high[k]) / 2);
      camera.distance = Math.max(1.5, Math.hypot(high[0] - low[0], high[1] - low[1]) * 1.2);
      draw();
    }

    /* The ramp spans the middle of the height distribution, not its extremes: one stray
     * point on a ceiling would otherwise flatten the whole floor to a single colour.
     * Percentiles off a sample, because sorting half a million floats to choose two
     * colours is not a good trade. */
    function rescaleHeights() {
      const used = counts.cloud * 3;
      const sample = [];
      for (let i = 2; i < used; i += 3 * 97) sample.push(cloudData[i]);
      sample.sort((a, b) => a - b);
      if (sample.length > 4) {
        heightRange = [sample[Math.floor(sample.length * 0.02)],
                       sample[Math.floor(sample.length * 0.98)]];
      }
    }

    /* Where Spot is right now, and which way it is pointing. The dot alone says where but
     * never which way, and which way is what tells an operator whether the wall filling
     * the view is the one ahead of the robot or the one behind it. */
    function setRobot(pose) {
      if (!pose || typeof pose.x !== "number") {
        counts.robot = 0;
        counts.heading = 0;
        return draw();
      }
      const at = [pose.x, pose.y, pose.z || 0];
      const yaw = pose.yaw || 0;
      const nose = [at[0] + Math.cos(yaw), at[1] + Math.sin(yaw), at[2]];
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.robot);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(at), gl.DYNAMIC_DRAW);
      counts.robot = 1;
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.heading);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([...at, ...nose]), gl.DYNAMIC_DRAW);
      counts.heading = 2;
      draw();
    }

    /* Put the camera over the robot without changing how far out or how steeply the
     * operator has it — following, rather than re-framing, is what makes a live view
     * watchable while Spot walks. */
    function followRobot(pose) {
      if (!pose || typeof pose.x !== "number") return;
      camera.target = [pose.x, pose.y, pose.z || 0];
      draw();
    }

    function setGraph(layout) {
      waypoints = layout.waypoints
        .filter((w) => w.x !== null && w.y !== null)
        .map((w) => ({ id: w.id, name: w.name, xyz: [w.x, w.y, w.z || 0] }));

      const byId = new Map(waypoints.map((w) => [w.id, w]));
      const lines = [];
      for (const edge of layout.edges) {
        const from = byId.get(edge.from);
        const to = byId.get(edge.to);
        if (!from || !to) continue;
        lines.push(...from.xyz, ...to.xyz);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.edges);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lines), gl.STATIC_DRAW);
      counts.edges = lines.length / 3;

      const marks = [];
      for (const waypoint of waypoints) marks.push(...waypoint.xyz);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.marks);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(marks), gl.STATIC_DRAW);
      counts.marks = marks.length / 3;

      const tags = [];
      for (const fiducial of layout.fiducials || []) tags.push(fiducial.x, fiducial.y, fiducial.z || 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffers.fiducials);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(tags), gl.STATIC_DRAW);
      counts.fiducials = tags.length / 3;

      frame();
    }

    function setSelection(next) {
      selection = next || { pick: "", place: "", home: "", patrol: [] };
      draw();
    }

    /* Put the whole map in view — the first thing anybody wants, and the thing they want
     * again after losing themselves in an orbit. */
    function frame() {
      if (!waypoints.length) return draw();
      const xs = waypoints.map((w) => w.xyz[0]);
      const ys = waypoints.map((w) => w.xyz[1]);
      const zs = waypoints.map((w) => w.xyz[2]);
      camera.target = [
        (Math.min(...xs) + Math.max(...xs)) / 2,
        (Math.min(...ys) + Math.max(...ys)) / 2,
        (Math.min(...zs) + Math.max(...zs)) / 2,
      ];
      const span = Math.max(
        Math.max(...xs) - Math.min(...xs),
        Math.max(...ys) - Math.min(...ys),
        4,
      );
      camera.distance = span * 1.3;
      camera.azimuth = -Math.PI / 4;
      camera.elevation = 0.95;
      draw();
    }

    /* ── pointer ──────────────────────────────────────────────────────────── */

    let dragging = null;
    let moved = 0;

    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      dragging = { x: event.clientX, y: event.clientY, pan: event.shiftKey || event.button === 2 };
      moved = 0;
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      const dx = event.clientX - dragging.x;
      const dy = event.clientY - dragging.y;
      dragging.x = event.clientX;
      dragging.y = event.clientY;
      moved += Math.abs(dx) + Math.abs(dy);

      if (dragging.pan) {
        // Pan across the ground plane, in the direction the camera is facing, scaled by
        // distance so the drag moves the same amount of *picture* at any zoom.
        const scale = camera.distance * 0.0016;
        const right = [Math.sin(camera.azimuth), -Math.cos(camera.azimuth), 0];
        const forward = [Math.cos(camera.azimuth), Math.sin(camera.azimuth), 0];
        camera.target[0] += (-dx * right[0] + dy * forward[0]) * scale;
        camera.target[1] += (-dx * right[1] + dy * forward[1]) * scale;
      } else {
        camera.azimuth -= dx * 0.006;
        camera.elevation = Math.max(-1.5, Math.min(1.5, camera.elevation + dy * 0.006));
      }
      draw();
    });

    canvas.addEventListener("pointerup", (event) => {
      const wasDragging = dragging;
      dragging = null;
      // A drag that ends over a waypoint is still a drag: only a click that stayed put
      // is asking about the thing under it.
      if (!wasDragging || moved > 6) return;
      const box = canvas.getBoundingClientRect();
      const at = { x: event.clientX - box.left, y: event.clientY - box.top };
      const hit = nearest(at);
      if (handlers.onPick) handlers.onPick(hit, { x: event.clientX, y: event.clientY });
    });

    canvas.addEventListener("contextmenu", (event) => event.preventDefault());

    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      camera.distance = Math.max(1, Math.min(400, camera.distance * (1 + Math.sign(event.deltaY) * 0.12)));
      draw();
    }, { passive: false });

    function nearest(at) {
      let best = null;
      for (const waypoint of screenPositions()) {
        const distance = Math.hypot(waypoint.screen.x - at.x, waypoint.screen.y - at.y);
        if (distance < 18 && (!best || distance < best.distance)) best = { ...waypoint, distance };
      }
      return best;
    }

    const observer = new ResizeObserver(() => draw());
    observer.observe(canvas);

    return {
      draw, frame, frameCloud, setCloud, appendCloud, setColorCloud, clearCloud, setGraph, setSelection,
      setRobot, followRobot, screenPositions, toScreen,
      get points() { return counts.cloud; },
      get selection() { return selection; },
    };
  }

  function link(gl, vertexSource, fragmentSource) {
    const program = gl.createProgram();
    for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, fragmentSource]]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader));
      }
      gl.attachShader(program, shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    return program;
  }

  return { mount };
})();
