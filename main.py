"""
2D real-time wind tunnel.

Drop an image into images/ (or pass a path as an argument, or drag-and-drop a
file onto the window while it's running) and it becomes the obstacle. Drag with
the mouse to pitch it live while the flow field - speed, pressure, or vorticity
- updates in real time, powered by a numba-JIT D2Q9 lattice-Boltzmann solver.
A live net-force arrow (drag + lift, via momentum exchange) shows how the
aerodynamic load changes as you pitch and reshape the object.

Run:
    python main.py                     auto-loads the first image in images/,
                                        or a built-in default shape if empty
    python main.py path/to/image.png   use a specific image

Controls:
    click + drag left/right    rotate the object (pitch)
    drag-and-drop an image     load it as the new obstacle, live
    left / right arrow keys    fine-tune angle by 2 degrees
    up / down arrow keys       increase / decrease inlet flow speed
    F                          mirror the object left-right
    A                          toggle the aerodynamic force arrow
    1 / 2 / 3                  view: speed / pressure / vorticity
    S                          toggle smoke streamlines
    [ / ]                      slow down / speed up smoke drift
    C                          save a screenshot to captures/
    R                          reset the flow
    space                      pause / resume
    esc / close window         quit
"""
import sys
import glob
import os
import time
import numpy as np
import pygame

import lbm
from obstacle import load_mask_from_image, rotate_mask, mirror_mask, default_shape_mask, place_mask
from colormap import apply_colormap

# ---------------------------------------------------------------- settings
NX, NY = 300, 100            # simulation grid (cells)
SCALE = 5                    # pixels per cell in the window
TAU = 0.56                   # relaxation time -> viscosity nu = (TAU-0.5)/3
INLET_U_INIT = 0.08           # initial inlet speed, lattice units
INLET_U_MIN, INLET_U_MAX = 0.02, 0.15
SUBSTEPS_PER_FRAME = 4        # physics steps per rendered frame
OBJ_SPAN = 34                  # object size in cells
MASK_CANVAS = int(OBJ_SPAN * 1.75)  # rotation-safe padded canvas (avoids clipping)
ANCHOR = (NX // 3, NY // 2)         # object position in the tunnel
N_PARTICLES = 500
PARTICLE_SPEED_MULT_INIT = 16.0
PARTICLE_SPEED_MIN, PARTICLE_SPEED_MAX = 3.0, 60.0
MAX_ANGLE = 180.0
DRAG_SENSITIVITY = 0.4        # degrees of rotation per pixel of mouse drag
RANGE_SMOOTHING = 0.85        # higher = color range adapts more slowly/smoothly
FORCE_SMOOTHING = 0.88        # higher = force arrow steadier (averages out shedding)
FORCE_ARROW_GAIN = 1500.0     # pixels per lattice-force-unit for the net arrow
FORCE_ARROW_MAX_PX = 230.0    # clamp so a strong force can't shoot off-screen
FORCE_DEADZONE = 0.002        # below this magnitude, don't draw the arrow
# ---------------------------------------------------------------------------

NET_COLOR = (255, 214, 10)      # net force arrow
DRAG_COLOR = (80, 210, 235)     # drag component
LIFT_COLOR = (235, 120, 220)    # lift component


def find_default_image():
    here = os.path.dirname(os.path.abspath(__file__))
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        found = sorted(glob.glob(os.path.join(here, "images", ext)))
        if found:
            return found[0]
    return None


def draw_arrow(surface, start, vec_px, color, width=3, head=10):
    """Draw a line-plus-triangle arrow from start along vec_px (pixels)."""
    x0, y0 = start
    x1, y1 = x0 + vec_px[0], y0 + vec_px[1]
    length = (vec_px[0] ** 2 + vec_px[1] ** 2) ** 0.5
    if length < 1.0:
        return
    pygame.draw.line(surface, color, (x0, y0), (x1, y1), width)
    ux, uy = vec_px[0] / length, vec_px[1] / length          # unit along arrow
    nx_, ny_ = -uy, ux                                        # perpendicular
    h = min(head, length * 0.5)
    left = (x1 - ux * h + nx_ * h * 0.55, y1 - uy * h + ny_ * h * 0.55)
    right = (x1 - ux * h - nx_ * h * 0.55, y1 - uy * h - ny_ * h * 0.55)
    pygame.draw.polygon(surface, color, [(x1, y1), left, right])


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else find_default_image()

    if image_path:
        print(f"Loading obstacle from: {image_path}")
        base_mask_img = load_mask_from_image(image_path, OBJ_SPAN, MASK_CANVAS)
    else:
        print("No image found - using a built-in default shape.")
        print("Drop a PNG/JPG into images/, pass a path, or drag-and-drop onto the window.")
        base_mask_img = default_shape_mask(OBJ_SPAN, MASK_CANVAS)

    pygame.init()
    pygame.display.set_caption("2D Wind Tunnel")
    screen = pygame.display.set_mode((NX * SCALE, NY * SCALE))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 20)

    angle = 0.0
    inlet_u = INLET_U_INIT
    particle_speed_mult = PARTICLE_SPEED_MULT_INIT
    view_mode = 1  # 1=speed 2=pressure 3=vorticity
    view_names = {1: "speed", 2: "pressure", 3: "vorticity"}
    show_smoke = True
    show_force = True
    paused = False
    dragging = False
    last_mouse_x = 0

    f = lbm.init_equilibrium(NX, NY, inlet_u)
    f_new = np.zeros_like(f)
    obstacle = place_mask((NX, NY), rotate_mask(base_mask_img, angle), *ANCHOR)

    force = np.zeros(2)              # scratch buffer written by lbm.step each call
    smoothed_force = np.zeros(2)     # displayed (drag, lift), EMA-smoothed

    # smoothed color-range state, per view, so the colormap doesn't jump frame-to-frame
    smoothed_range = {1: [0.0, inlet_u * 1.5], 2: [1e-3], 3: [1e-3]}

    rng = np.random.default_rng(0)
    px = np.zeros(N_PARTICLES)
    py = np.zeros(N_PARTICLES)

    def reset_particles(mask=None):
        if mask is None:
            mask = np.ones(N_PARTICLES, dtype=bool)
        n = int(mask.sum())
        px[mask] = rng.uniform(0, 3, n)
        py[mask] = rng.uniform(2, NY - 2, n)

    reset_particles()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.DROPFILE:
                try:
                    new_mask = load_mask_from_image(event.file, OBJ_SPAN, MASK_CANVAS)
                    base_mask_img = new_mask
                    f = lbm.init_equilibrium(NX, NY, inlet_u)
                    smoothed_force[:] = 0.0
                    reset_particles()
                    print(f"loaded obstacle from: {event.file}")
                except Exception as e:
                    print(f"couldn't load '{event.file}': {e}")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    f = lbm.init_equilibrium(NX, NY, inlet_u)
                    smoothed_force[:] = 0.0
                    reset_particles()
                elif event.key == pygame.K_1:
                    view_mode = 1
                elif event.key == pygame.K_2:
                    view_mode = 2
                elif event.key == pygame.K_3:
                    view_mode = 3
                elif event.key == pygame.K_s:
                    show_smoke = not show_smoke
                elif event.key == pygame.K_a:
                    show_force = not show_force
                elif event.key == pygame.K_f:
                    base_mask_img = mirror_mask(base_mask_img)
                elif event.key == pygame.K_LEFT:
                    angle -= 2.0
                elif event.key == pygame.K_RIGHT:
                    angle += 2.0
                elif event.key == pygame.K_UP:
                    inlet_u = min(INLET_U_MAX, inlet_u + 0.005)
                elif event.key == pygame.K_DOWN:
                    inlet_u = max(INLET_U_MIN, inlet_u - 0.005)
                elif event.key == pygame.K_LEFTBRACKET:
                    particle_speed_mult = max(PARTICLE_SPEED_MIN, particle_speed_mult * 0.8)
                elif event.key == pygame.K_RIGHTBRACKET:
                    particle_speed_mult = min(PARTICLE_SPEED_MAX, particle_speed_mult * 1.25)
                elif event.key == pygame.K_c:
                    os.makedirs("captures", exist_ok=True)
                    fname = os.path.join("captures", f"wind_tunnel_{int(time.time())}.png")
                    pygame.image.save(screen, fname)
                    print(f"saved {fname}")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                last_mouse_x = event.pos[0]
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - last_mouse_x
                last_mouse_x = event.pos[0]
                angle += dx * DRAG_SENSITIVITY

        angle = max(-MAX_ANGLE, min(MAX_ANGLE, angle))

        rotated = rotate_mask(base_mask_img, angle)
        obstacle = place_mask((NX, NY), rotated, *ANCHOR)

        if not paused:
            fx_sum = 0.0
            fy_sum = 0.0
            for _ in range(SUBSTEPS_PER_FRAME):
                lbm.step(f, f_new, obstacle, TAU, inlet_u, force)
                f, f_new = f_new, f
                fx_sum += force[0]
                fy_sum += force[1]
            inv = 1.0 / SUBSTEPS_PER_FRAME
            smoothed_force[0] = FORCE_SMOOTHING * smoothed_force[0] + (1 - FORCE_SMOOTHING) * fx_sum * inv
            smoothed_force[1] = FORCE_SMOOTHING * smoothed_force[1] + (1 - FORCE_SMOOTHING) * fy_sum * inv

            rho, ux, uy = lbm.macroscopic(f, obstacle)

            if np.isnan(rho).any():
                print("Simulation went unstable - resetting. "
                      "Try lowering inlet speed (down arrow) or raising TAU in main.py.")
                f = lbm.init_equilibrium(NX, NY, inlet_u)
                smoothed_force[:] = 0.0
                rho, ux, uy = lbm.macroscopic(f, obstacle)

            if show_smoke:
                ix = np.clip(px.astype(np.int32), 0, NX - 1)
                iy = np.clip(py.astype(np.int32), 0, NY - 1)
                px += ux[ix, iy] * particle_speed_mult
                py += uy[ix, iy] * particle_speed_mult
                out = (px >= NX - 2) | (px < 0) | (py < 1) | (py >= NY - 1) | obstacle[ix, iy]
                if out.any():
                    reset_particles(out)
        else:
            rho, ux, uy = lbm.macroscopic(f, obstacle)

        # --------------------------------------------------------- render
        valid = ~obstacle

        if view_mode == 1:
            field = np.sqrt(ux ** 2 + uy ** 2)
            target_vmax = max(np.percentile(field[valid], 99), inlet_u * 0.3)
            s = smoothed_range[1]
            s[1] = RANGE_SMOOTHING * s[1] + (1 - RANGE_SMOOTHING) * target_vmax
            vmin, vmax = 0.0, s[1]
            rgb = apply_colormap(field, vmin=vmin, vmax=vmax, style="speed")
        elif view_mode == 2:
            baseline = rho[valid].mean()
            dev = rho - baseline
            target_scale = max(np.percentile(np.abs(dev[valid]), 99), 1e-3)
            s = smoothed_range[2]
            s[0] = RANGE_SMOOTHING * s[0] + (1 - RANGE_SMOOTHING) * target_scale
            vmin, vmax = -s[0], s[0]
            rgb = apply_colormap(dev, vmin=vmin, vmax=vmax, style="diverging")
        else:
            vort = lbm.vorticity(ux, uy, obstacle)
            target_scale = max(np.percentile(np.abs(vort[valid]), 99), 1e-3)
            s = smoothed_range[3]
            s[0] = RANGE_SMOOTHING * s[0] + (1 - RANGE_SMOOTHING) * target_scale
            vmin, vmax = -s[0], s[0]
            rgb = apply_colormap(vort, vmin=vmin, vmax=vmax, style="diverging")

        rgb[obstacle] = (45, 45, 50)

        surf = pygame.surfarray.make_surface(rgb)
        surf = pygame.transform.scale(surf, (NX * SCALE, NY * SCALE))
        screen.blit(surf, (0, 0))

        if show_smoke:
            for x, y in zip(px, py):
                xi, yi = int(x), int(y)
                if 0 <= xi < NX and 0 <= yi < NY and not obstacle[xi, yi]:
                    pygame.draw.circle(screen, (255, 255, 255), (int(x * SCALE), int(y * SCALE)), 1)

        # ---- aerodynamic force arrow (net = drag + lift), same axes as the flow ----
        drag, lift = smoothed_force[0], smoothed_force[1]
        mag = (drag ** 2 + lift ** 2) ** 0.5
        if show_force and mag > FORCE_DEADZONE:
            cx = ANCHOR[0] * SCALE
            cy = ANCHOR[1] * SCALE
            scale_px = FORCE_ARROW_GAIN
            if mag * scale_px > FORCE_ARROW_MAX_PX:
                scale_px = FORCE_ARROW_MAX_PX / mag       # clamp overall length
            # faint component arrows (drag = horizontal, lift = vertical)
            draw_arrow(screen, (cx, cy), (drag * scale_px, 0.0), DRAG_COLOR, width=2, head=8)
            draw_arrow(screen, (cx, cy), (0.0, lift * scale_px), LIFT_COLOR, width=2, head=8)
            # bold net arrow on top
            draw_arrow(screen, (cx, cy), (drag * scale_px, lift * scale_px), NET_COLOR, width=4, head=13)

        ld_ratio = abs(lift) / drag if drag > 1e-6 else 0.0
        hud = [
            f"angle {angle:+.1f}deg   view: {view_names[view_mode]}   "
            f"smoke: {'on' if show_smoke else 'off'}   force: {'on' if show_force else 'off'}"
            f"{'   PAUSED' if paused else ''}",
            f"inlet_u {inlet_u:.3f}   smoke_speed {particle_speed_mult:.0f}   "
            f"drag {drag:+.4f}  lift {lift:+.4f}  L/D {ld_ratio:.2f}  (arb. units)",
            "drag=rotate | drop img to load | up/dn=speed | [ ]=smoke | F=flip | A=force | 1/2/3=view | C=capture | R=reset | space=pause",
        ]
        for i, text in enumerate(hud):
            screen.blit(font.render(text, True, (255, 255, 255), (0, 0, 0)), (6, 6 + i * 18))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
