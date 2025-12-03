#!/usr/bin/env python3
"""
mrt_point_write_hdr.py

Create an equal-area 360×180 HDR panorama of a simple rectangular room
from a given camera position and compute MRT [°C] at that point.

Surfaces:
  - E: east wall  (x = 0)
  - W: west wall  (x = room_w)
  - N: north wall (z = 0)
  - S: south wall (z = room_d)
  - ceiling: y = room_h
  - floor:   y = 0

All surface values are in °C (equivalent temperatures). The HDR image
stores °C in all three channels (monochrome).

MRT is computed from the ray-cast panorama using Stefan–Boltzmann:
  T_i (°C) → T_i (K) → E_i = eps * sigma * T_i^4
Then:
  E_avg = mean(E_i over all rays that hit a surface)
  MRT   = (E_avg / (eps * sigma))^(1/4)

Example (explicit geometry):

    python mrt_point_write_hdr.py \\
        --room_w 3.55 --room_d 4.90 --room_h 3.00 \\
        --cam_x 1.775 --cam_y 1.30 --cam_z 1.60 \\
        --TE 25.0 --TW 26.0 --TS 25.5 --TN 21.0 \\
        --Tceiling 24.5 --Tfloor 23.5 \\
        --eps 1.0 \\
        --out mrt_equal-area.hdr

Example using a profile file (profile.cfg):

    room_w=3.55
    room_d=4.90
    room_h=3.00
    cam_x=1.775
    cam_y=1.30
    cam_z=1.60

    python mrt_point_write_hdr.py \\
        --profile profile.cfg \\
        --TE 25.0 --TW 26.0 --TS 25.5 --TN 21.0 \\
        --Tceiling 24.5 --Tfloor 23.5 \\
        --eps 1.0 \\
        --out mrt_equal-area.hdr
"""

import argparse
import math

import numpy as np
import cv2

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

DEFAULT_CAM_X = DEFAULT_ROOM_W / 2.0
DEFAULT_CAM_Y = 1.3
DEFAULT_CAM_Z = 1.6

# Width of panorama; height is computed as W / pi (equal-area)
DEFAULT_IMG_W = 1256


# -------------------------------------------------------------------
# Profile loading
# -------------------------------------------------------------------
def load_profile(path):
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
# CLI parsing
# -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Create an equal-area HDR panorama from room surface temperatures "
            "and compute MRT at the camera position."
        )
    )

    # Geometry / camera via profile + optional CLI overrides
    p.add_argument("--profile", help="Profile .cfg with room_w, room_d, room_h, cam_x, cam_y, cam_z")

    p.add_argument("--room_w", type=float, help="Room width (east→west, x-axis)")
    p.add_argument("--room_d", type=float, help="Room depth (north→south, z-axis)")
    p.add_argument("--room_h", type=float, help="Room height (floor→ceiling, y-axis)")

    p.add_argument("--cam_x", type=float, help="Camera x position")
    p.add_argument("--cam_y", type=float, help="Camera y position (height)")
    p.add_argument("--cam_z", type=float, help="Camera z position")

    # Image size
    p.add_argument(
        "--img_w",
        type=int,
        default=DEFAULT_IMG_W,
        help="Panorama width in pixels (default: 1256). "
             "Height is computed as round(width / pi)."
    )

    # Surface temperatures (required, °C)
    p.add_argument("--TE", type=float, required=True, help="Temperature for east wall (x=0) in °C")
    p.add_argument("--TW", type=float, required=True, help="Temperature for west wall (x=room_w) in °C")
    p.add_argument("--TS", type=float, required=True, help="Temperature for south wall (z=room_d) in °C")
    p.add_argument("--TN", type=float, required=True, help="Temperature for north wall (z=0) in °C")
    p.add_argument("--Tceiling", type=float, required=True, help="Temperature for ceiling (y=room_h) in °C")
    p.add_argument("--Tfloor", type=float, required=True, help="Temperature for floor (y=0) in °C")

    # Emissivity (for MRT)
    p.add_argument(
        "--eps",
        type=float,
        default=1.0,
        help="Emissivity ε (default: 1.0) used for MRT computation",
    )

    # Output
    p.add_argument("-o", "--out", required=True, help="Output HDR filename")

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
    # 1) Geometry + camera
    # -----------------------------
    geom = {
        "room_w": DEFAULT_ROOM_W,
        "room_d": DEFAULT_ROOM_D,
        "room_h": DEFAULT_ROOM_H,
        "cam_x":  DEFAULT_CAM_X,
        "cam_y":  DEFAULT_CAM_Y,
        "cam_z":  DEFAULT_CAM_Z,
    }

    # Apply profile if given
    if args.profile:
        prof = load_profile(args.profile)
        for k in geom:
            if k in prof:
                geom[k] = prof[k]

    # Override via CLI if provided
    for key in ["room_w", "room_d", "room_h", "cam_x", "cam_y", "cam_z"]:
        val = getattr(args, key)
        if val is not None:
            geom[key] = val

    room_w = geom["room_w"]
    room_d = geom["room_d"]
    room_h = geom["room_h"]
    cam_x  = geom["cam_x"]
    cam_y  = geom["cam_y"]
    cam_z  = geom["cam_z"]

    print("Using geometry + camera:")
    print(f"  room_w={room_w:.3f}, room_d={room_d:.3f}, room_h={room_h:.3f}")
    print(f"  cam_x={cam_x:.3f}, cam_y={cam_y:.3f}, cam_z={cam_z:.3f}")

    # -----------------------------
    # 2) Equal-area image size
    # -----------------------------
    img_w = args.img_w
    img_h = int(round(img_w / math.pi))  # Lambert cylindrical equal-area aspect
    print(f"Image size: {img_w} x {img_h} (equal-area)")

    # -----------------------------
    # 3) Surface temperatures (°C)
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
    # 4) Ray casting (Lambert cylindrical equal-area)
    #    + MRT accumulation from ray-cast pixels
    # -----------------------------
    img = np.zeros((img_h, img_w, 3), dtype=np.float32)

    ox, oy, oz = cam_x, cam_y, cam_z

    E_sum = 0.0       # sum of irradiances from all hit pixels
    hit_count = 0     # number of pixels whose rays hit a surface

    for j in range(img_h):
        if j % 40 == 0 or j == img_h - 1:
            print(f"  row {j + 1}/{img_h}")

        v = (j + 0.5) / img_h
        # Lambert cylindrical equal-area:
        #   sin(phi) = 1 - 2v  → phi in [-pi/2, +pi/2]
        sin_phi = 1.0 - 2.0 * v
        sin_phi = max(-1.0, min(1.0, sin_phi))
        phi = math.asin(sin_phi)
        cos_phi = math.cos(phi)

        for i in range(img_w):
            u = (i + 0.5) / img_w
            # full 360°:
            #   u=0.0 → -180°
            #   u=0.5 →   0° (forward, +z)
            #   u=1.0 → +180°
            yaw = (u - 0.5) * 2.0 * math.pi

            sin_yaw = math.sin(yaw)
            cos_yaw = math.cos(yaw)

            # direction
            dx = cos_phi * sin_yaw   # left-right
            dy = sin_phi             # up-down
            dz = cos_phi * cos_yaw   # front-back (+z forward)

            t_min = float("inf")
            hit_surface = None

            # East wall E: x = 0
            t = intersect_plane(
                coord_origin=ox, coord_dir=dx, fixed_value=0.0,
                other_origin1=oy, other_dir1=dy, min1=0.0, max1=room_h,
                other_origin2=oz, other_dir2=dz, min2=0.0, max2=room_d,
            )
            if t is not None and t < t_min:
                t_min = t
                hit_surface = "E"

            # West wall W: x = room_w
            t = intersect_plane(
                coord_origin=ox, coord_dir=dx, fixed_value=room_w,
                other_origin1=oy, other_dir1=dy, min1=0.0, max1=room_h,
                other_origin2=oz, other_dir2=dz, min2=0.0, max2=room_d,
            )
            if t is not None and t < t_min:
                t_min = t
                hit_surface = "W"

            # North wall N: z = 0
            t = intersect_plane(
                coord_origin=oz, coord_dir=dz, fixed_value=0.0,
                other_origin1=ox, other_dir1=dx, min1=0.0, max1=room_w,
                other_origin2=oy, other_dir2=dy, min2=0.0, max2=room_h,
            )
            if t is not None and t < t_min:
                t_min = t
                hit_surface = "N"

            # South wall S: z = room_d
            t = intersect_plane(
                coord_origin=oz, coord_dir=dz, fixed_value=room_d,
                other_origin1=ox, other_dir1=dx, min1=0.0, max1=room_w,
                other_origin2=oy, other_dir2=dy, min2=0.0, max2=room_h,
            )
            if t is not None and t < t_min:
                t_min = t
                hit_surface = "S"

            # Floor: y = 0
            t = intersect_plane(
                coord_origin=oy, coord_dir=dy, fixed_value=0.0,
                other_origin1=ox, other_dir1=dx, min1=0.0, max1=room_w,
                other_origin2=oz, other_dir2=dz, min2=0.0, max2=room_d,
            )
            if t is not None and t < t_min:
                t_min = t
                hit_surface = "floor"

            # Ceiling: y = room_h
            t = intersect_plane(
                coord_origin=oy, coord_dir=dy, fixed_value=room_h,
                other_origin1=ox, other_dir1=dx, min1=0.0, max1=room_w,
                other_origin2=oz, other_dir2=dz, min2=0.0, max2=room_d,
            )
            if t is not None and t < t_min:
                t_min = t
                hit_surface = "ceiling"

            if hit_surface is None:
                # Ray did not hit the room: keep 0°C in HDR, and do NOT
                # include it in MRT (no contribution).
                T_C = 0.0
            else:
                T_C = temps_C[hit_surface]

                # Accumulate irradiance for MRT (from this direction)
                T_K = T_C + 273.15
                E_pixel = eps * SIGMA * (T_K ** 4)
                E_sum += E_pixel
                hit_count += 1

            # store same scalar (°C) in R,G,B
            img[j, i, :] = T_C

    # -----------------------------
    # 5) Compute MRT from ray-cast image
    # -----------------------------
    if hit_count == 0:
        print("\nWARNING: No rays hit any surface; MRT cannot be computed.")
        MRT_K = float("nan")
        MRT_C = float("nan")
        E_avg = float("nan")
    else:
        E_avg = E_sum / hit_count
        MRT_K = (E_avg / (eps * SIGMA)) ** 0.25
        MRT_C = MRT_K - 273.15

        print("\n=== MRT from ray-cast equal-area panorama ===")
        print(f"  Emissivity eps   = {eps:.3f}")
        print(f"  Hits / pixels    = {hit_count} / {img_w * img_h}")
        print(f"  E_avg (W/m^2)    = {E_avg:.6f}")
        print(f"  MRT_K (Kelvin)   = {MRT_K:.6f} K")
        print(f"  MRT_C (Celsius)  = {MRT_C:.6f} °C")
        print("=============================================\n")

    # -----------------------------
    # 6) Write HDR
    # -----------------------------
    out_name = args.out
    img_bgr = img[..., ::-1]  # RGB → BGR for OpenCV
    ok = cv2.imwrite(out_name, img_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write {out_name}")

    print(f"Saved HDR: {out_name}")
    print(f"  shape: {img.shape}, dtype: {img.dtype}")
    print(f"  value min/max in image: {img[..., 0].min():.3f} {img[..., 0].max():.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()