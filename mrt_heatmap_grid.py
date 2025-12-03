"""
mrt_heatmap_grid.py

Author: Dong Hyun Kim (in collaboration with GPT‑5.1 on 3 December 2025)

Compute a 2D heatmap of MRT [°C] as a function of camera position (cam_x, cam_z)
at a fixed cam_y, using the same ray-cast MRT logic as
visualize_equalarea_and_calculate_mrt_at_defined_camera_position.py.

For each grid point (x, z) inside the room, we:
  - cast rays in all directions (Lambert cylindrical equal-area sampling),
  - for each ray, find which surface it hits,
  - convert the hit surface temperature T_i (°C) → T_i (K),
  - compute E_i = eps * sigma * T_i^4,
  - average E_i over all rays that hit surfaces,
  - convert E_avg back to MRT: T_MRT = (E_avg / (eps * sigma))^(1/4).

The result is a 2D array MRT[z_idx, x_idx] (°C), visualized as a heatmap:

  x-axis: 0 → room_w (east → west)
  y-axis: 0 → room_d (north → south)

Usage example:

    python mrt_heatmap_grid.py \\
        --profile profile.cfg \\
        --cam_y 1.30 \\
        --TE 25 --TW 26 --TS 25.5 --TN 21 \\
        --Tceiling 24.5 --Tfloor 23.5 \\
        --eps 1.0 \\
        --grid_step 0.1 \\
        --rays_w 360 --rays_h 180 \\
        --out_png mrt_heatmap.png \\
        --out_npy mrt_values.npy

Notes:
  - All surface temperatures are in °C.
  - Emissivity eps is assumed equal for all surfaces (default 1.0).
  - Ray resolution (rays_w, rays_h) controls angular sampling quality vs speed.
"""

import argparse
import math

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Physical constant
# -------------------------------------------------------------------
SIGMA = 5.670374419e-8  # W·m^-2·K^-4

# -------------------------------------------------------------------
# Defaults for room + camera (can be overridden by profile or CLI)
# -------------------------------------------------------------------
DEFAULT_ROOM_W = 3.55
DEFAULT_ROOM_D = 4.90
DEFAULT_ROOM_H = 3.00

DEFAULT_CAM_Y = 1.3  # default measurement height


# -------------------------------------------------------------------
# Profile loading
# -------------------------------------------------------------------
def load_profile(path: str):
    """
    Load a simple key=value config file.

    Lines starting with '#' or empty lines are ignored.
    Keys and values are stripped; values are parsed as float.
    """
    cfg = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            try:
                cfg[key] = float(val)
            except ValueError:
                # Ignore non-float values silently
                continue
    return cfg


# -------------------------------------------------------------------
# Plane intersection helper
# -------------------------------------------------------------------
def intersect_plane(coord_origin, coord_dir,
                    fixed_value,
                    other_origin1, other_dir1, min1, max1,
                    other_origin2, other_dir2, min2, max2):
    """
    Generic helper for a plane like x = const, y = const, or z = const.

    Returns t (float) if the ray hits inside the bounds, else None.
    """
    if abs(coord_dir) < 1e-9:
        return None  # ray parallel to plane

    t = (fixed_value - coord_origin) / coord_dir
    if t <= 0:
        return None  # hit is behind camera

    p1 = other_origin1 + t * other_dir1
    p2 = other_origin2 + t * other_dir2

    if (min1 - 1e-9 <= p1 <= max1 + 1e-9) and (min2 - 1e-9 <= p2 <= max2 + 1e-9):
        return t
    return None


# -------------------------------------------------------------------
# MRT at a single point
# -------------------------------------------------------------------
def compute_mrt_for_position(
    cam_x: float,
    cam_y: float,
    cam_z: float,
    room_w: float,
    room_d: float,
    room_h: float,
    temps_C: dict,
    eps: float,
    rays_w: int,
    rays_h: int,
):
    """
    Compute MRT [°C] from ray-cast sampling at a single camera position.

    temps_C: dict with keys "E", "W", "S", "N", "ceiling", "floor"
             containing temperatures in °C.
    """
    ox, oy, oz = cam_x, cam_y, cam_z

    E_sum = 0.0
    hit_count = 0

    # Lambert cylindrical equal-area sampling over the sphere.
    for j in range(rays_h):
        v = (j + 0.5) / rays_h
        # sin(phi) = 1 - 2v → phi in [-pi/2, +pi/2]
        sin_phi = 1.0 - 2.0 * v
        sin_phi = max(-1.0, min(1.0, sin_phi))
        phi = math.asin(sin_phi)
        cos_phi = math.cos(phi)

        for i in range(rays_w):
            u = (i + 0.5) / rays_w
            # full 360°:
            #   u=0.0 → -180°
            #   u=0.5 →   0° (forward, +z)
            #   u=1.0 → +180°
            yaw = (u - 0.5) * 2.0 * math.pi

            sin_yaw = math.sin(yaw)
            cos_yaw = math.cos(yaw)

            dx = cos_phi * sin_yaw   # left-right
            dy = sin_phi             # up-down
            dz = cos_phi * cos_yaw   # front-back (+z forward)

            t_min = float("inf")
            hit_surface = None

            # East wall E: x = 0
            t = intersect_plane(
                coord_origin=ox, coord_dir=dx, fixed_value=0.0,
                other_origin1=oy, other_dir1=dy, min1=0.0,    max1=room_h,
                other_origin2=oz, other_dir2=dz, min2=0.0,    max2=room_d,
            )
            if t is not None and t < t_min:
                t_min = t
                hit_surface = "E"

            # West wall W: x = room_w
            t = intersect_plane(
                coord_origin=ox, coord_dir=dx, fixed_value=room_w,
                other_origin1=oy, other_dir1=dy, min1=0.0,    max1=room_h,
                other_origin2=oz, other_dir2=dz, min2=0.0,    max2=room_d,
            )
            if t is not None and t < t_min:
                t_min = t
                hit_surface = "W"

            # North wall N: z = 0
            t = intersect_plane(
                coord_origin=oz, coord_dir=dz, fixed_value=0.0,
                other_origin1=ox, other_dir1=dx, min1=0.0,    max1=room_w,
                other_origin2=oy, other_dir2=dy, min2=0.0,    max2=room_h,
            )
            if t is not None and t < t_min:
                t_min = t
                hit_surface = "N"

            # South wall S: z = room_d
            t = intersect_plane(
                coord_origin=oz, coord_dir=dz, fixed_value=room_d,
                other_origin1=ox, other_dir1=dx, min1=0.0,    max1=room_w,
                other_origin2=oy, other_dir2=dy, min2=0.0,    max2=room_h,
            )
            if t is not None and t < t_min:
                t_min = t
                hit_surface = "S"

            # Floor: y = 0
            t = intersect_plane(
                coord_origin=oy, coord_dir=dy, fixed_value=0.0,
                other_origin1=ox, other_dir1=dx, min1=0.0,    max1=room_w,
                other_origin2=oz, other_dir2=dz, min2=0.0,    max2=room_d,
            )
            if t is not None and t < t_min:
                t_min = t
                hit_surface = "floor"

            # Ceiling: y = room_h
            t = intersect_plane(
                coord_origin=oy, coord_dir=dy, fixed_value=room_h,
                other_origin1=ox, other_dir1=dx, min1=0.0,    max1=room_w,
                other_origin2=oz, other_dir2=dz, min2=0.0,    max2=room_d,
            )
            if t is not None and t < t_min:
                t_min = t
                hit_surface = "ceiling"

            if hit_surface is None:
                continue  # ray escapes; no contribution

            T_C = temps_C[hit_surface]
            T_K = T_C + 273.15
            E_pixel = eps * SIGMA * (T_K ** 4)
            E_sum += E_pixel
            hit_count += 1

    if hit_count == 0:
        return math.nan

    E_avg = E_sum / hit_count
    MRT_K = (E_avg / (eps * SIGMA)) ** 0.25
    MRT_C = MRT_K - 273.15
    return MRT_C


# -------------------------------------------------------------------
# CLI parsing
# -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compute an MRT heatmap over cam_x–cam_z positions at fixed cam_y, "
            "using ray-cast MRT from surface temperatures."
        )
    )

    # Geometry / camera via profile + optional CLI overrides
    p.add_argument("--profile", help="Profile .cfg with room_w, room_d, room_h, cam_y (optional)")

    p.add_argument("--room_w", type=float, help="Room width (east→west, x-axis)")
    p.add_argument("--room_d", type=float, help="Room depth (north→south, z-axis)")
    p.add_argument("--room_h", type=float, help="Room height (floor→ceiling, y-axis)")

    p.add_argument("--cam_y", type=float, help="Camera height (y coordinate). If not given, use profile or default.")

    # Grid over cam_x and cam_z
    p.add_argument("--grid_step", type=float, default=0.1,
                   help="Grid step for cam_x and cam_z in meters (default: 0.1)")

    # Ray sampling resolution
    p.add_argument("--rays_w", type=int, default=360,
                   help="Number of rays in horizontal direction (default: 360)")
    p.add_argument("--rays_h", type=int, default=180,
                   help="Number of rays in vertical direction (default: 180)")

    # Surface temperatures (required, °C)
    p.add_argument("--TE", type=float, required=True, help="Temperature for east wall (x=0) in °C")
    p.add_argument("--TW", type=float, required=True, help="Temperature for west wall (x=room_w) in °C")
    p.add_argument("--TS", type=float, required=True, help="Temperature for south wall (z=room_d) in °C")
    p.add_argument("--TN", type=float, required=True, help="Temperature for north wall (z=0) in °C")
    p.add_argument("--Tceiling", type=float, required=True, help="Temperature for ceiling (y=room_h) in °C")
    p.add_argument("--Tfloor", type=float, required=True, help="Temperature for floor (y=0) in °C")

    # Emissivity (for MRT)
    p.add_argument("--eps", type=float, default=1.0,
                   help="Emissivity ε (default: 1.0) used for MRT computation")

    # Output files
    p.add_argument("--out_png", required=True,
                   help="Output heatmap PNG filename")
    p.add_argument("--out_npy", default=None,
                   help="Optional output .npy file for MRT array")

    return p.parse_args()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    args = parse_args()

    if not (0.0 < args.eps <= 1.0):
        raise SystemExit("Emissivity (--eps) must be in (0,1].")

    eps = args.eps

    # -----------------------------
    # 1) Geometry + cam_y
    # -----------------------------
    geom = {
        "room_w": DEFAULT_ROOM_W,
        "room_d": DEFAULT_ROOM_D,
        "room_h": DEFAULT_ROOM_H,
        "cam_y":  DEFAULT_CAM_Y,
    }

    # Apply profile if given
    if args.profile:
        prof = load_profile(args.profile)
        for k in geom:
            if k in prof:
                geom[k] = prof[k]

    # Override via CLI if provided
    for key in ["room_w", "room_d", "room_h", "cam_y"]:
        val = getattr(args, key)
        if val is not None:
            geom[key] = val

    room_w = geom["room_w"]
    room_d = geom["room_d"]
    room_h = geom["room_h"]
    cam_y  = geom["cam_y"]

    print("Using geometry:")
    print(f"  room_w={room_w:.3f}, room_d={room_d:.3f}, room_h={room_h:.3f}")
    print(f"Constant cam_y = {cam_y:.3f} m")

    # -----------------------------
    # 2) Surface temperatures (°C)
    # -----------------------------
    temps_C = {
        "E":       args.TE,
        "W":       args.TW,
        "S":       args.TS,
        "N":       args.TN,
        "ceiling": args.Tceiling,
        "floor":   args.Tfloor,
    }

    print("Surface temperatures (°C):")
    for k, v in temps_C.items():
        print(f"  {k:8s} = {v:.6f} C")

    # -----------------------------
    # 3) Grid over cam_x, cam_z
    # -----------------------------
    step = args.grid_step
    if step <= 0:
        raise SystemExit("grid_step must be > 0")

    x_vals = np.arange(0.0, room_w + 1e-9, step)
    z_vals = np.arange(0.0, room_d + 1e-9, step)

    nx = len(x_vals)
    nz = len(z_vals)

    print(f"\nGrid: {nx} points in x (0..{room_w:.3f}), {nz} points in z (0..{room_d:.3f})")
    print(f"Total grid points: {nx * nz}")
    print(f"Ray sampling: {args.rays_w} × {args.rays_h}\n")

    mrt_map = np.full((nz, nx), np.nan, dtype=float)

    # -----------------------------
    # 4) Loop over grid points
    # -----------------------------
    for iz, z in enumerate(z_vals):
        print(f"Row {iz+1}/{nz}  (cam_z = {z:.3f} m)")
        for ix, x in enumerate(x_vals):
            MRT_C = compute_mrt_for_position(
                cam_x=x,
                cam_y=cam_y,
                cam_z=z,
                room_w=room_w,
                room_d=room_d,
                room_h=room_h,
                temps_C=temps_C,
                eps=eps,
                rays_w=args.rays_w,
                rays_h=args.rays_h,
            )
            mrt_map[iz, ix] = MRT_C

    # -----------------------------
    # 5) Save MRT array (optional)
    # -----------------------------
    if args.out_npy is not None:
        np.save(args.out_npy, mrt_map)
        print(f"\nSaved MRT array to {args.out_npy} (shape={mrt_map.shape})")

    # -----------------------------
    # 6) Plot heatmap
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        mrt_map,
        extent=[0, room_w, 0, room_d],
        origin="lower",
        aspect="auto",
        cmap="inferno",
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("MRT (°C)")

    ax.set_xlabel("cam_x (m)   [E (x=0)  →  W (x=room_w)]")
    ax.set_ylabel("cam_z (m)   [N (z=0)  →  S (z=room_d)]")
    ax.set_title(f"MRT heatmap at cam_y = {cam_y:.2f} m")

    # Wall labels (E, W, N, S)
    ax.text(0.01 * room_w, 0.5 * room_d, "E", color="white",
            fontsize=12, ha="left", va="center", weight="bold")
    ax.text(0.99 * room_w, 0.5 * room_d, "W", color="white",
            fontsize=12, ha="right", va="center", weight="bold")
    ax.text(0.5 * room_w, 0.01 * room_d, "N", color="white",
            fontsize=12, ha="center", va="bottom", weight="bold")
    ax.text(0.5 * room_w, 0.99 * room_d, "S", color="white",
            fontsize=12, ha="center", va="top", weight="bold")

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Saved heatmap PNG to {args.out_png}")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
