Tools to compute and visualise mean radiant temperature (MRT) in a simple rectangular room.

Assumptions:
- Rectangular room:
  - `room_w` – width (x, east–west)
  - `room_d` – depth (z, north–south)
  - `room_h` – height (y, floor–ceiling)
- Uniform radiant temperature per surface.
- Surfaces:
  - E – east wall  (`x = 0`)
  - W – west wall  (`x = room_w`)
  - N – north wall (`z = 0`)
  - S – south wall (`z = room_d`)
  - ceiling (`y = room_h`)
  - floor   (`y = 0`)
- Temperatures in °C, emissivity `eps` shared by all surfaces.

## Scripts

- **`mrt_point_write_hdr.py`**
  - Computes MRT at a single camera position.
  - Ray-casts to all room surfaces (E/W/N/S/ceiling/floor).
  - Writes an equal-area 360×180 HDR panorama with temperature in each pixel.
  - Prints MRT in Kelvin and °C to the terminal.

- **`mrt_heatmap_grid.py`**
  - Computes MRT on a horizontal grid (cam_x–cam_z) at fixed `cam_y`.
  - Uses the same ray-casting logic as `mrt_point_write_hdr.py`.
  - Outputs:
    - 2D MRT array (`.npy`),
    - plan-view heatmap image with E/W/N/S labelled.
