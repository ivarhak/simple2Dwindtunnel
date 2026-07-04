"""
D2Q9 Lattice-Boltzmann solver, JIT-compiled with numba for real-time performance.

Lattice layout (index: direction, (cx,cy)):
    0: rest   (0, 0)      5: NE (1, 1)
    1: E      (1, 0)      6: NW (-1, 1)
    2: N      (0, 1)      7: SW (-1,-1)
    3: W      (-1, 0)     8: SE (1,-1)
    4: S      (0,-1)
"""
import numpy as np
from numba import njit

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


@njit(cache=True, fastmath=True, inline="always")
def _feq(rho, ux, uy, k):
    cu = CX[k] * ux + CY[k] * uy
    usqr = ux * ux + uy * uy
    return W[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr)


def init_equilibrium(nx, ny, inlet_u):
    """Initialize the whole domain at rest-ish equilibrium with the inlet velocity."""
    f = np.zeros((nx, ny, 9), dtype=np.float64)
    for k in range(9):
        cu = CX[k] * inlet_u
        f[:, :, k] = W[k] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * inlet_u * inlet_u)
    return f


@njit(cache=True, fastmath=True)
def step(f, f_new, obstacle, tau, inlet_u, force_out, max_speed=0.3):
    """
    Advance the simulation by one LBM timestep.
    Mutates `f` in place (turns it into the post-collision state) and writes
    the fully streamed next state into `f_new`. Caller is responsible for
    swapping (f, f_new) = (f_new, f) after calling this.

    force_out: preallocated shape-(2,) float64 array, overwritten each call
    with the (Fx, Fy) net force exerted BY the fluid ON the obstacle in
    lattice units, via the momentum-exchange method (summed over every
    bounce-back link touching the obstacle, excluding the tunnel walls).
    Because the freestream is always +x regardless of the object's pitch,
    Fx is drag and Fy is lift directly, with no extra projection needed.

    Boundary conditions:
      - obstacle cells: full bounce-back (no-slip walls)
      - top/bottom domain edges: bounce-back (tunnel walls)
      - left edge (i=0): fixed-velocity inlet
      - right edge (i=nx-1): open (fixed-density) outlet
    """
    nx, ny, _ = f.shape
    force_out[0] = 0.0
    force_out[1] = 0.0

    # ---- 1. macroscopic variables + BGK collision (in place into f) ----
    for i in range(nx):
        for j in range(ny):
            if obstacle[i, j]:
                continue
            rho = 0.0
            ux = 0.0
            uy = 0.0
            for k in range(9):
                fk = f[i, j, k]
                rho += fk
                ux += fk * CX[k]
                uy += fk * CY[k]
            if rho > 1e-9:
                ux /= rho
                uy /= rho
            else:
                rho, ux, uy = 1.0, 0.0, 0.0

            speed = (ux * ux + uy * uy) ** 0.5
            if speed > max_speed:
                scale = max_speed / speed
                ux *= scale
                uy *= scale

            for k in range(9):
                feq = _feq(rho, ux, uy, k)
                f[i, j, k] -= (f[i, j, k] - feq) / tau

    # ---- 2. streaming + bounce-back off obstacle / top / bottom walls ----
    f_new[:, :, :] = 0.0
    for i in range(nx):
        for j in range(ny):
            if obstacle[i, j]:
                continue  # solid cells never emit into the fluid
            for k in range(9):
                ni = i + CX[k]
                nj = j + CY[k]
                if 0 <= ni < nx and obstacle[ni, nj]:
                    # bounce-back off the object: reverse the population AND
                    # tally the momentum handed to the obstacle (2 * c_k * f_k)
                    fk = f[i, j, k]
                    f_new[i, j, OPPOSITE[k]] += fk
                    force_out[0] += 2.0 * CX[k] * fk
                    force_out[1] += 2.0 * CY[k] * fk
                elif nj < 0 or nj >= ny:
                    # bounce-back off the tunnel's top/bottom walls (no force tally)
                    f_new[i, j, OPPOSITE[k]] += f[i, j, k]
                elif 0 <= ni < nx:
                    f_new[ni, nj, k] += f[i, j, k]
                # else: ni out of x-range -> leaves domain (fixed up by inlet/outlet below)

    # ---- 3. inlet: fixed-velocity Dirichlet BC on column 0 ----
    for j in range(ny):
        if not obstacle[0, j]:
            for k in range(9):
                f_new[0, j, k] = _feq(1.0, inlet_u, 0.0, k)

    # ---- 4. outlet: open (fixed-density) outflow on last column ----
    # Pinning density to the same reference as the inlet (rather than a raw
    # zero-gradient copy) stops the domain's mean density/pressure from
    # drifting upward over long runs - velocity is still extrapolated from
    # the interior, so flow rate responds naturally to blockage/angle.
    for j in range(ny):
        if obstacle[nx - 1, j]:
            continue
        rho2 = 0.0
        ux2 = 0.0
        uy2 = 0.0
        for k in range(9):
            fk = f_new[nx - 2, j, k]
            rho2 += fk
            ux2 += fk * CX[k]
            uy2 += fk * CY[k]
        if rho2 > 1e-9:
            ux2 /= rho2
            uy2 /= rho2
        for k in range(9):
            f_new[nx - 1, j, k] = _feq(1.0, ux2, uy2, k)

    # ---- 5. reset obstacle interior to rest equilibrium ----
    # (matters when the mask rotates: newly-exposed fluid cells start clean)
    for i in range(nx):
        for j in range(ny):
            if obstacle[i, j]:
                for k in range(9):
                    f_new[i, j, k] = _feq(1.0, 0.0, 0.0, k)


@njit(cache=True, fastmath=True)
def macroscopic(f, obstacle):
    nx, ny, _ = f.shape
    rho = np.ones((nx, ny))
    ux = np.zeros((nx, ny))
    uy = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            if obstacle[i, j]:
                continue
            r = 0.0
            u = 0.0
            v = 0.0
            for k in range(9):
                fk = f[i, j, k]
                r += fk
                u += fk * CX[k]
                v += fk * CY[k]
            rho[i, j] = r
            if r > 1e-9:
                ux[i, j] = u / r
                uy[i, j] = v / r
    return rho, ux, uy


@njit(cache=True, fastmath=True)
def vorticity(ux, uy, obstacle):
    nx, ny = ux.shape
    vort = np.zeros((nx, ny))
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            if obstacle[i, j]:
                continue
            dvdx = uy[i + 1, j] - uy[i - 1, j]
            dudy = ux[i, j + 1] - ux[i, j - 1]
            vort[i, j] = dvdx - dudy
    return vort
