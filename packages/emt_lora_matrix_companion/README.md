> **Status: GUI assets only — no LoRA implementation exists in this repository.**
> This package contains mockups and a layout spec for a future adapter-selection
> UI. There is no adapter training code, no adapter weight format, and no runtime
> adapter path in `src/` or `training/`; `grep -ri lora` over the source tree
> returns only this package and its asset zip. Treat "runtime LoRA" as *planned*.
> Per-preset merged weights (separate `.bin` files) are the mechanism the format
> supports today and are the simpler, faster ship path — see
> [`../../ARCHITECTURE_FIDELITY.md`](../../ARCHITECTURE_FIDELITY.md) §1.2 and §5.

# EmT 16-LoRA Matrix Companion - GUI Asset Package

A companion neural LoRA selection and morphing plugin GUI.

## Included Assets:
1. `master_gui_1200x600.png` - Full composite 1200x600 companion interface with 4x4 matrix and dual morph dials.
2. `lora_sigil_tile_active.png` - Transparent backlit sacred geometry LoRA tile (molten amber glow).
3. `lora_sigil_tile_inactive.png` - Transparent dormant sacred geometry LoRA tile (smoked amber / stone relief).
4. `dial_calibration_ring.png` - Transparent circular calibration track for the morph controls.
5. `dial_cap_monolith.png` - Transparent turned metal rotary knob cap with matte tangerine indicator notch.

## Layout Specs:
- Canvas: 1200 x 600 px
- Matrix: 4x4 grid of 16 LoRA talisman buttons (80x80 px each)
- Dials: Left (LoRA Blend Weight), Right (SSM Delta / Morph)
- Color Palette: Deep black ash (#0D0E11), dormant amber (#281A10), molten amber glow (#FFA834 / #E66819).
