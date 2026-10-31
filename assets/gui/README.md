# GUI mockups

Concept images for a future GUI. The plugin does not load any of these files;
the current editor is the X11/Win32 code in `src/clap/gui_*.cpp`.

| File | What it is |
|---|---|
| `emt_instrument_monolith.png` | Full 1200×600 instrument mockup: 4 dials (Gain, Attack, Release, Mix) and bypass |
| `chassis_background.png` | Blank 1200×600 faceplate |
| `knob_cap_monolith*.png`, `knob_calibration_ring*.png` | Dial cap and ring, opaque and transparent |
| `toggle_bypass_active*.png` | Bypass switch |
| `matrix_companion_mockup.png`, `matrix_tile_active.png`, `matrix_tile_inactive.png` | Mockup for a 4×4 preset/adapter matrix. Nothing in `src/` or `training/` implements adapters; per-preset weight files are what the format supports today |
| `preview_synth_ui.png`, `preview_synth_4knobs.png` | Earlier full-UI previews |

Palette: #0D0E11 background, #281A10 dormant amber, #E8732A tangerine.
