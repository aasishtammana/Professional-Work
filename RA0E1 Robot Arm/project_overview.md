## Executive Summary
This repository combines an embedded Renesas RA0E1 firmware example (I2C sensor + UART/RTT) with a PC-side DexArm portrait-drawing pipeline (webcam capture → sketch → SVG → G-code → serial). It demonstrates end-to-end integration from sensor-level embedded code to robotic plotting on the desktop.

## Architecture Overview
```
[PC / Windows]
  OpenCV webcam -> sketch (OpenCV) -> vectorize (contours or potrace) -> SVG
  SVG -> parse/scale -> G-code -> Serial(COMx, 115200) -> DexArm

[Renesas RA0E1]
  BSP clocks -> FSP drivers (I2C, UART) -> VL6180X init/read -> UART/RTT logs
```
Key directories:
- `RA0E1_Robot_Arm/src`: application modules (`hal_entry.c`, `i2c_sensor.c/.h`, `sau_uart_ep.c/.h`, `SEGGER_RTT/*`)
- `RA0E1_Robot_Arm/ra_gen`: FSP-generated instances/configuration
- Root Python scripts: `version1.py`, `version2.py`, `svg_generator.py`

## Data and Control Flow
- Face capture: `detect_face()` saves `detected_face.jpg` on user keypress.
- Sketching: `generate_darker_sketch()` produces `sketch_output.jpg`.
- Vectorization: either contours (`svgwrite`) or potrace creates `Final_line_sketch.svg`.
- Toolpath: SVG parsing → bounds → uniform scaling/centering → transformed paths.
- Motion: G-code generation + serial streaming to DexArm with homing before/after.
- Embedded: I2C sensor polling → UART/RTT logs; buttons generate short/long press events.

## Dependencies
- Python (PC): opencv-python, numpy, pillow, svgwrite, svg.path, pyserial, potrace (bindings)
- Embedded (RA0E1): Renesas FSP drivers configured in `ra_gen/`; SEGGER RTT middleware

## System Requirements
- PC: Windows, webcam, DexArm connected on a COM port (e.g., `COM4`), Python 3.9+
- Embedded: RA0E1 Fast Prototyping Board, VL6180X (I2C, 3.3V), J-Link (e2 studio or Renesas Flash Programmer)

## Setup and Quick Start
- DexArm (smoother potrace variant):
  1. `pip install opencv-python numpy pillow svgwrite svg.path pyserial potrace` (bindings)
  2. Connect DexArm; set `dexarm_port` (e.g., `COM4`).
  3. Tune `z_up`/`z_down` and draw window.
  4. Run `version2.py` and follow prompts.
- Firmware flashing:
  - Use Renesas Flash Programmer; select `RA0E1_Robot_Arm/Debug/RA0E1_Robot_Arm.hex`; interface SWD via J-Link; press Start.

## Production Considerations
- Robustness: add serial timeouts/retries; handle DexArm error codes; validate camera availability; fallback image.
- Safety: always home (`G28`), add E-stop, ensure Z limits are safe for pen mount.
- Reproducibility: freeze package versions in `requirements.txt`; document COM port and Z calibration.
- Performance: keep 512×512 raster for controlled complexity; reduce Bezier interpolation points to shrink G-code.

## Future Enhancements
- Add CLI flags for COM port, speeds, scaling window, vectorizer choice.
- Add path optimization beyond greedy nearest-neighbor (e.g., 2-opt) to reduce travel.
- Render a preview of transformed paths over a workspace outline.
- Stream G-code with chunked flow control and progress reporting.
- RA0E1: add UART RX command handler, DMA for TX, and structured frames with CRC.
