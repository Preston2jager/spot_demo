# Spot — RMIT Centre for Future Construction

Boston Dynamics Spot control: segmentation and classification on the robot's own Jetson,
GraphNav navigation, a sorting demo, laptop-side 3D reconstruction, and a one-page browser console.

```bash
uv sync
source .venv/bin/activate
python start.py      # http://localhost:8080
```

Nothing moves the robot until you press something on the page — connecting is not
standing, and opening the page does not take Spot's lease.

## The console (8080)

The left column, top to bottom: **Stop**, the **AI vision** switch, then three blocks —
**System** (*Status Check*, *Initialisation*), **Posture** (walk with WASD, *Stand*, *Sit*) and
**Actions** (grasp here, sort, patrol, go home). Everything else is the display area:

| tab | |
|---|---|
| Hand Camera | the gripper camera |
| Body Cameras | the body cameras as a 2×2 grid: front pair, back, left, right |
| Depth Camera | the CoreIO RealSense, colour and depth blended with a slider (all the way left is colour, right is depth); click to measure a distance |
| Map, Live | the GraphNav map Spot holds, in 3D, and the live depth stitch |
| 3D Recon | a frontal reconstruction fused **on this laptop** from the hand camera, the front pair and the RealSense |
| Log | the run log; warnings that arrive elsewhere are counted on the tab |

**AI** is a layer over Hand Camera and Body Cameras: every segmented object stays in colour on
a grey scene, outlined by bin, labelled with confidence and grip material; `?` marks an
uncertain class. Open `http://…:8080/#cameras` to land on a tab.

**Status Check** is the soft reset: time sync, lease, behaviour faults, E-Stop, system faults,
battery and the CoreIO services, checked and tidied — nothing moves and nothing plays. Every
command runs it first. **Initialisation** is the full one: stand, announce, report, sit, power off.

## Stop

The red button at the top of the command column, or **Esc**. It freezes Spot where it is —
a stop command, not a sit and not a stand — and nothing a mission tries afterwards reaches
the robot until you dispatch something new. It works with or without a mission running.
Errors inside a mission do the same thing: hold position and report.

## Stance

*Stand* on a standing robot and *Sit* on a sitting one do nothing and say so. The console
asks the robot rather than trusting what it last did.

## The sorting demo

1. A freshly started Spot holds no map. If the **Map** tab says so, *Load … onto Spot*
   uploads `SPOT_MAP_DIR` and localises (park Spot facing a fiducial).
2. Press **Sort — you choose** or **Sort — automatic**. A window opens over the map: choose the
   **pick point** and the **Teddy bear**, **Bottle** and **Other** bins (or click waypoints), then
   **Start**. It refuses until every point is set and on Spot's map. Patrol and Go Home work
   the same way.
   * **you choose**: at the pick point Spot scans with the body cameras and you click the
     object to take — on Body Cameras or, switching tab, on a Hand Camera scan. The menu offers
     all three bins with the predicted one first. *Rescan* looks again.
   * **Sort — automatic**: the same loop, taking the most confident object each time.
3. Spot walks to the object, lines up on it (see below), grasps, grips it at the torque its
   material calls for, carries it to the bin, drops it in, and returns to the pick point.

**Registered objects.** Spot does not decide from one frame: each grasp mission scans a few
times, and an object counts only once it has been seen in the same place in at least two scans.
You choose from the ten best, each outlined in every camera that sees it (dashed where this scan
missed it and only its registered position is shown). Labels are what the AI calls the object;
the colour is the bin it sorts into.

**Lining up.** Spot no longer grasps from where it first saw the object. It walks to a
standoff, then every half second aims the hand camera at the object, finds it again,
re-measures it with depth, and turns and steps until the object is straight ahead at grasping
range. Only then does the manipulation API grasp, without walking again.

## Running things on the robot

```bash
python coreio/deploy_vision.py      # the vision service, as the spot-vision Dockge stack
python coreio/deploy_realsense.py   # the RealSense bridge (redeploy for 3D Recon's /rgbd)
python coreio/deploy_web.py         # the console itself, as the demo_web container
```

To change what the demo sorts, edit `coreio/vision/vocab/sort3.json`, re-bake and redeploy —
see `coreio/README.md`.

## Recording a map

1. Record with `src/ref/recording_command_line.py` — a fiducial must be in view at the start.
2. Download the map into `runtime/maps/<name>/` and point `SPOT_MAP_DIR` at it.

## Configuration

Copy `.env.example` to `.env`. Robot credentials go in `credentials.json` (or
`SPOT_HOSTNAME` / `SPOT_USERNAME` / `SPOT_PASSWORD`). `CLAUDE.md` is the architecture write-up.
