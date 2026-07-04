<img width="1440" height="518" alt="Screenshot 2026-07-04 at 5 33 52 PM" src="https://github.com/user-attachments/assets/0c774038-db1b-413d-af1b-b07a03f3a5eb" />

# 2D Wind Tunnel

Drop an image in, drag to pitch it, and watch a real-time fluid simulation
flow around it - speed, pressure, or vorticity, your choice - with a live
net-force arrow showing how drag and lift respond as you pitch and reshape
the object. Powered by a numba-JIT-compiled D2Q9 lattice-Boltzmann solver
(~200+ physics steps/sec on a laptop CPU at the default grid size, so live
rotation stays smooth). You can drag-and-drop a new image onto the window
any time without restarting.

## Setup

```
pip install -r requirements.txt
```

(If that fails with an externally-managed-environment error on Linux, add
`--break-system-packages`, or use a virtualenv.)

If that fails just install everything manually, modules needed are listed in requirements.txt, make sure you are using a python version that supports all modules. :P

## Run

```
python main.py                     # auto-loads the first image in images/,
                                    # or a built-in default shape if empty
python main.py path/to/image.png   # use a specific image
```

The first launch takes a few extra seconds while numba compiles the solver;
it's cached after that, so subsequent launches start fast.

## Importing your own image

**Easiest path:** drop a PNG or JPG into the `images/` folder and just run
`python main.py` - no code editing, no arguments needed. It auto-picks the
first image it finds there. You can also pass a path directly as an
argument, which overrides the `images/` folder.

**While it's running:** just drag any image file from your file manager and
drop it onto the window - it loads immediately as the new obstacle and the
flow restarts around it. No need to quit and relaunch.

**Best format: PNG with a transparent background.** Export or crop your
object (a CAD silhouette, logo, part outline, airfoil profile) so the
background is transparent alpha, not just white or black. The alpha channel
is used directly as the obstacle shape, so there's no ambiguity about what
counts as "object" vs "background" - this gives the cleanest, most reliable
mask.

**Also works: a photo or PNG/JPG on a plain background**, light or dark. The
script converts to grayscale and thresholds against the average brightness
to guess the silhouette, auto-detecting whether the object or the background
is the darker region. This works well for a clear, high-contrast, simple
silhouette against a flat background, but a busy or low-contrast photo will
give a noisy or wrong mask - if the shape looks off in the tunnel, that's
usually why. Transparent PNG avoids this entirely.

The image is resized to fit the tunnel automatically - any resolution or
aspect ratio works.

## Controls

| Action | Effect |
|---|---|
| Click + drag left/right | Rotate the object (pitch), live |
| Drag-and-drop an image | Load it as the new obstacle, live |
| Left / Right arrow keys | Fine-tune angle by 2 degrees |
| Up / Down arrow keys | Increase / decrease inlet flow speed |
| F | Mirror the object left-right (fixes objects facing the wrong way) |
| A | Toggle the aerodynamic force arrow |
| 1 / 2 / 3 | Switch view: speed / pressure / vorticity |
| S | Toggle smoke streamlines on/off |
| [ / ] | Slow down / speed up the smoke drift |
| C | Save a screenshot to `captures/` |
| R | Reset the flow |
| Space | Pause / resume |
| Esc or close window | Quit |

## The force arrow

The yellow arrow anchored on the object is the **net aerodynamic force** the
fluid exerts on it, computed by the momentum-exchange method (summing the
momentum flipped at every bounce-back point on the object's surface each
step). Because the freestream always points along +x, the force splits
cleanly into two components, drawn as faint arrows:

- **Drag** (cyan, horizontal) - force along the flow direction. Always
  positive/downstream; grows as you pitch the object and present more
  frontal area.
- **Lift** (magenta, vertical) - force perpendicular to the flow. Near zero
  for a symmetric shape at zero pitch, and flips sign as you pitch up vs
  down. The HUD also shows the lift-to-drag ratio (L/D), a standard
  aerodynamic efficiency measure.

The arrow is heavily smoothed over time so it stays readable even when the
instantaneous force oscillates (which it does whenever vortices are shedding
off the object). Toggle it with **A**.

**This is deliberately non-quantitative.** The numbers are in arbitrary
lattice units, not newtons, and the arrow length is capped so a strong force
can't shoot off-screen - so read it for *relative* trends (how lift and drag
change as you pitch, mirror, or reshape the object), not for absolute values.
The directions and sign relationships are physically correct (verified
against Newton's third law: the force on the object is equal and opposite to
the momentum imparted to the fluid), but the magnitudes shouldn't be treated
as calibrated measurements.

## Tuning

All the knobs are constants near the top of `main.py`:

- `TAU` - relaxation time. Controls viscosity: `nu = (TAU - 0.5)/3`. Lower
  values (closer to 0.5) mean less viscous flow and more dramatic vortex
  shedding, but can go numerically unstable on sharp/thin shapes at high
  angle. Default `0.56` is tuned to be stable across most silhouettes.
- `INLET_U` - freestream speed. Higher = faster-looking flow and higher
  effective Reynolds number, but also closer to the instability edge.
- `NX, NY` - grid resolution. `300x100` leaves a large real-time performance
  margin on a laptop CPU (measured ~300 fps physics-only at the default 4
  substeps/frame); you can raise it for a crisper image if your machine has
  room, or lower it for an even bigger performance cushion.
- `SUBSTEPS_PER_FRAME` - physics steps computed per rendered frame. More
  substeps = flow reacts faster to rotation changes; there's a lot of
  headroom here before frame rate becomes a concern.
- `OBJ_SPAN` - the object's size in grid cells.

Speed, pressure, and vorticity colors are all auto-scaled each frame (a
smoothed 99th-percentile of the current field), so the color range adapts
to whatever object/angle/speed you're looking at instead of washing out or
saturating to one color on shapes the original fixed range didn't suit.

## limitations

- This is a qualitative/educational simulator, not a validated CFD tool -
  fine for seeing stagnation points, flow separation, and vortex shedding
  develop believably, not for reading off precise lift/drag numbers. The
  force arrow shows correct directions and relative trends, but its
  magnitudes are in arbitrary units (see "The force arrow" above).
- Very thin, sharp, or highly asymmetric shapes at extreme angles can push
  the solver toward instability (it auto-resets if that happens - you'll
  see a message in the console). Lowering `INLET_U` or raising `TAU`
  slightly gives more headroom.
- It's 2D - this shows a cross-sectional slice of flow, not full 3D
  aerodynamics.


Credits:
Vibe coded with the help of Claude by Antropic
Feel free to use and customize however you want
Ivar Haakansson 2026
