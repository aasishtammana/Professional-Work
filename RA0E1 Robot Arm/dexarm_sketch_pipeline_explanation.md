## Executive Summary
The DexArm sketch pipeline captures a face from a webcam, converts it into a pencil-style raster sketch, vectorizes it to SVG (two methods), transforms the coordinates into the DexArm workspace, and streams motion as G-code over serial. Two end-to-end variants (`version1.py`, `version2.py`) differ in their vectorization approach (contour simplification vs. potrace Bezier tracing). `svg_generator.py` provides the raster-to-SVG step in isolation for debugging.

## Architecture and Data Flow
```
Camera → detect_face → sketch_output.jpg → vectorize (contours or potrace) → Final_line_sketch.svg
SVG → parse_path → bounds → uniform scaling + centering → transform_paths
Transformed SVG paths → G-code generation (pen-up/down, feeds) → Serial → DexArm
```
Artifacts:
- `detected_face.jpg`: cropped face image
- `sketch_output.jpg`: darker pencil-like raster
- `Final_line_sketch.svg`: vector representation

## Dependencies
- OpenCV (`cv2`), NumPy (`numpy`)
- Pillow (`PIL.Image`) and `svgwrite` for contour SVG path writing (v1)
- `potrace` Python bindings for bitmap-to-Bezier tracing (v2)
- `svg.path` and `xml.dom.minidom` for SVG parsing
- `pyserial` for DexArm communication

## DexArm Portrait Sketch Pipeline – Unified Explanation

This document consolidates the functionality of `svg_generator.py`, `version1.py`, and `version2.py` into one explanation. It describes the end-to-end pipeline used to detect a face, generate a line-art sketch, convert to SVG, and command a DexArm to draw via G-code over serial.

### High-Level Workflow
1. Capture face photo via webcam with OpenCV and Haar cascade (`detect_face`).
2. Convert the cropped face into a darker pencil-like sketch (`generate_darker_sketch`).
3. Vectorize the sketch to SVG:
   - Approach A (version1): Edge detect → contour simplify → SVG path lines (using `svgwrite`).
   - Approach B (version2): Adaptive threshold → `potrace` bitmap trace → SVG cubic paths.
4. Parse SVG paths and scale/translate to DexArm coordinate window.
5. Generate G-code toolpaths with pen-up/pen-down and feedrates; stream over serial.

---

### Modules and Functions

#### Face capture: `detect_face()`
- Uses `cv2.VideoCapture(0)` and `haarcascade_frontalface_default.xml` to detect faces.
- UI: Live preview with instructions. Press 's' to save `detected_face.jpg`; 'q' to exit.
- Cropping: Expands detected box by ~30% in both axes for aesthetic framing.
- Edge cases: No camera, no face detected; exits gracefully and releases camera.
```python
def detect_face():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    try:
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    px, py = int(0.3 * w), int(0.3 * h)
                    face_img = frame[max(0,y-py):y+h+py, max(0,x-px):x+w+px]
                    cv2.imwrite('detected_face.jpg', face_img)
                break
            elif key == ord('q'):
                break
    finally:
        cap.release(); cv2.destroyAllWindows()
```

#### Sketch generation: `generate_darker_sketch(image_path, output_path, scale, alpha, beta)`
- Pipeline: grayscale → invert → Gaussian blur → divide blend → contrast/brightness adjust.
- Parameters:
  - `scale` for `cv2.divide` normalization.
  - `alpha` (gain) and `beta` (bias) to darken strokes; tuned around `alpha≈1.35`, `beta≈-30`.
- Output: Saves `sketch_output.jpg`; returns the sketch array.
- Risks: Over-darkening may merge features; parameters depend on lighting.
```python
def generate_darker_sketch(image_path, output_path, scale=128, alpha=1.35, beta=-30):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv_gray = 255 - gray
    blurred = cv2.GaussianBlur(inv_gray, (23, 23), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=scale)
    darker = cv2.convertScaleAbs(sketch, alpha=alpha, beta=beta)
    cv2.imwrite(output_path, darker)
    return darker
```

#### SVG generation approaches
- `svg_generator.py`: `jpg_to_line_sketch_svg(input_path, output_path, threshold, resize_dim, epsilon, min_contour_length)`
  - Threshold to binary, Canny edges, `findContours`, simplify via `cv2.approxPolyDP` with `epsilon`.
  - Emits polyline paths with `svgwrite`. Controls fidelity via `epsilon` and `min_contour_length`.
  - Performance: Linear in number of edge pixels and contour length.
```python
def jpg_to_line_sketch_svg(input_path, output_path, threshold=128, resize_dim=(512,512), epsilon=5.0, min_contour_length=40):
    image = Image.open(input_path).convert('L').resize(resize_dim)
    binary = np.where(np.array(image) > threshold, 255, 0).astype(np.uint8)
    edges = cv2.Canny(binary, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dwg = svgwrite.Drawing(output_path, profile='tiny')
    for contour in contours:
        if len(contour) >= min_contour_length:
            pts = cv2.approxPolyDP(contour[:,0,:].astype(np.float32), epsilon, closed=False).reshape(-1,2)
            path = 'M ' + ' L '.join(f'{x},{y}' for x,y in pts)
            dwg.add(dwg.path(d=path, stroke='black', fill='none'))
    dwg.save()
```
- `version2.py`: `potrace` vectorization
  - Adaptive threshold to bitmap; traced with `potrace.Bitmap(...).trace()` to bezier curves.
  - Generates a single `<svg>` with `<path>` elements using cubic segments.
  - Produces smoother curves and fewer nodes; good for plotting.
```python
def jpg_to_line_sketch_svg(input_path, output_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (512,512))
    blurred = cv2.GaussianBlur(image, (7,7), 0)
    bitmap = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 125, 20) > 0
    path = potrace.Bitmap(bitmap).trace()
    def conv(p): return f"{p.x:.1f},{p.y:.1f}"
    svg_paths = []
    for curve in path:
        parts = [f"M{conv(curve.start_point)}"]
        for seg in curve.segments:
            parts.append(f"L{conv(seg.c)};L{conv(seg.end_point)}" if seg.is_corner else f"C{conv(seg.c1)} {conv(seg.c2)} {conv(seg.end_point)}")
        parts.append('Z'); svg_paths.append(' '.join(parts))
    with open(output_path, 'w') as f:
        f.write(f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512'>{''.join(f'<path d=\"{p}\" fill=\"black\"/>' for p in svg_paths)}</svg>")
```

#### SVG parsing and normalization
- `extract_paths(svg_file)`: Parses `<path>` elements (and `<rect>` fallback) using `xml.dom.minidom` and `svg.path.parse_path`.
- `get_svg_bounds(svg_file)`: Computes min/max over all segment endpoints.
- `calculate_uniform_scaling(svg_bounds, x_min, y_min, x_max, y_max, additional_scale)`: Uniformly scales and centers SVG into a target draw window.
- `transform_paths(paths, scale, offset_x, offset_y)`: Applies affine transform to complex coordinates of segments and greedily reorders paths to reduce travel.
```python
def calculate_uniform_scaling(bounds, x_min, y_min, x_max, y_max, s=1.0):
    min_x, min_y, max_x, max_y = bounds
    scale = min((x_max-x_min)/(max_x-min_x), (y_max-y_min)/(max_y-min_y)) * s
    off_x = x_min - min_x*scale + (x_max-x_min - (max_x-min_x)*scale)/2
    off_y = y_min - min_y*scale + (y_max-y_min - (max_y-min_y)*scale)/2
    return scale, off_x, off_y

def transform_paths(paths, scale, off_x, off_y):
    for path in paths:
        for seg in path:
            for attr in ['start','end','control1','control2']:
                if hasattr(seg, attr):
                    p = getattr(seg, attr)
                    setattr(seg, attr, complex(p.real*scale + off_x, p.imag*scale + off_y))
    ordered, current, remaining = [], complex(0,0), paths[:]
    while remaining:
        nxt = min(remaining, key=lambda p: abs(current - p[0].start))
        ordered.append(nxt); current = nxt[-1].end; remaining.remove(nxt)
    return ordered
```

#### G-code generation and streaming
- `svg_to_gcode(paths, z_up, z_down, draw_speed, move_speed, num_points)`
  - Lifts pen, rapid moves (`G0`) to start, lowers pen (`G1 Z...`), then feeds along segments.
  - For cubic segments, interpolates points (`num_points` per segment) using Bezier equation; for line segments, uses endpoint.
  - Parameters tuned per version:
    - v1: `z_up=-30`, `z_down=-41`, speeds `4000`.
    - v2: `z_up=-42`, `z_down=-48`.
```python
def svg_to_gcode(paths, z_up=-42, z_down=-47, draw_speed=2000, move_speed=2000, num_points=50):
    cmds = [f"G0 Z{z_up+5} F{move_speed}"]
    for path in paths:
        start = path[0].start
        cmds += [f"G0 X{start.real:.2f} Y{start.imag:.2f} Z{z_up} F{move_speed}", f"G1 Z{z_down} F{move_speed}"]
        for seg in path:
            if isinstance(seg, CubicBezier):
                for t in np.linspace(0,1,num_points):
                    p = (1-t)**3*seg.start + 3*(1-t)**2*t*seg.control1 + 3*(1-t)*t**2*seg.control2 + t**3*seg.end
                    cmds.append(f"G1 X{p.real:.2f} Y{p.imag:.2f} F{draw_speed}")
            else:
                cmds.append(f"G1 X{seg.end.real:.2f} Y{seg.end.imag:.2f} F{draw_speed}")
        cmds.append(f"G0 Z{z_up} F{move_speed}")
    return cmds
```
- `send_gcode(cmd)`: Writes line + waits for `ok`/`ready` response before proceeding.
```python
def send_gcode(command):
    ser.write((command + '\n').encode('utf-8'))
    while True:
        response = ser.readline().decode('utf-8').strip()
        if response.lower() in ['ok', 'ready']:
            break
```

---

### End-to-End Usage
- Connect DexArm over serial (e.g., `COM4` at `115200` baud).
- Run the pipeline variant:
  - `version1.py`: Camera → sketch → contour SVG → draw.
  - `version2.py`: Camera → sketch → potrace SVG → draw.
  - `svg_generator.py`: Only the raster-to-SVG step; helpful for debugging the vectorization.
- Adjust drawing window and Z positions:
  - Example window: `x_min=-125, y_min=165, x_max=125, y_max=355`.
  - Tune `additional_scale` to fit within paper and margins.

---

### Inter-dependencies
- `cv2`, `numpy`, `Pillow` (v1), `svgwrite` (v1), `potrace` Python bindings (v2), `svg.path`, `pyserial`, `xml.dom.minidom`.
- Output files used across steps: `detected_face.jpg` → `sketch_output.jpg` → `Final_line_sketch.svg`.

### Performance Notes
- Camera capture and cascade detection are real-time; ensure sufficient lighting.
- Vectorization complexity depends on edges (v1) or bitmap size (v2). 512×512 is a good balance.
- G-code size grows with curve interpolation; decrease `num_points` if streaming becomes slow.

### Risks & Gotchas
- Coordinate mismatches: SVG origin top-left vs DexArm XY plane; consistent transforms are applied.
- Serial flow control: Always wait for `ok`/`ready`; add timeouts for robustness.
- Z calibration: `z_up`/`z_down` must be tuned to your pen mount; test on scrap paper first.
- Face not found: Prompt instructs to press 's'. Consider adding a fallback to save full frame.

### Validation Checklist
- [ ] Webcam accessible; cascade file found; `detected_face.jpg` saved when pressing 's'
- [ ] `sketch_output.jpg` has clear edges; parameters produce visible lines
- [ ] `Final_line_sketch.svg` opens and contains `<path d="...">`
- [ ] Transformed paths fit within `[x_min,x_max]×[y_min,y_max]`; no clipping
- [ ] DexArm responds with `ok`/`ready`; pen lifts and lowers correctly; homing works
