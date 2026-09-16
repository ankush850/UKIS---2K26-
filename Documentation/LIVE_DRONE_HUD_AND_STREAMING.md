# Live Drone HUD Simulation & Real-Time Sortie Streaming Engine
## 60-Frame Optical Trajectory, WebSocket Telemetry & DGCA Meteorological Constraints

**Project Identifier:** NETRA-D (UKIS-2026 Problem P-008)  
**Problem Owner:** Disaster Mitigation and Management Centre (DMMC), Department of Disaster Management, Government of Uttarakhand  
**Component:** Netra Aerial Tactical Live Video Stream (`backend/aerial/live_stream.py` & `/api/aerial/live-stream/ws`)  
**Target Capabilities:** Tactical UAV Sortie Simulation, Real-Time PyTorch Frame Inference, DGCA Weather Compliance  

---

## 1. Operational Overview & Rationale

In high-stakes emergency command rooms, evaluating static drone photos can be slow. Incident commanders require a continuous, dynamic situational picture—monitoring tactical UAV sorties as they bank through mountain valleys, scan roads, and project live hazard telemetry.

However, during live hackathon evaluations and indoor disaster drills, flying a physical drone inside a conference hall or emergency briefing room is physically impossible and violates safety regulations.

### The NETRA-D Solution:
NETRA-D features a **Live Tactical Drone HUD Sortie Simulator**:
- Operators can upload any static aerial survey photograph (or select a pre-loaded Himalayan disaster preset).
- The engine synthesizes a smooth **60-frame Ken Burns optical flight trajectory**, simulating altitude variations, optical panning, and banking angles.
- **PyTorch deep learning inference runs live on every single video frame**, detecting floodwater and debris in real time.
- Flight telemetry (altitude, air speed, wind speed, hazard percentages, pitch/roll) is rendered onto an interactive aerospace HUD canvas via high-speed WebSockets.

```
+--------------------------------------------------------------------------------------------------+
|                            LIVE DRONE HUD STREAMING PIPELINE                                     |
+--------------------------------------------------------------------------------------------------+
|   Static Drone Orthomosaic / Pre-Recorded Sortie Video                                           |
|                              |                                                                   |
|                              v                                                                   |
|   60-Frame Ken Burns Trajectory Generator:                                                        |
|   - Smooth Pan: (Δx(t), Δy(t)) via Sinusoidal Interpolation                                      |
|   - Altitude Shift: h(t) = 80m -> 120m AGL                                                       |
|   - Banking Angle: θ(t) in [-8°, +8°]                                                            |
|                              |                                                                   |
|                              v                                                                   |
|   Live Frame Extractor (60 Discrete Frames at 15-20 FPS)                                         |
|                              |                                                                   |
|                              v                                                                   |
|   Real-Time PyTorch Inference Engine (Per-Frame Forward Pass):                                   |
|   - Multi-Hazard Semantic Segmentation (Floodwater, Mudflows, Debris)                            |
|   - Dynamic Bounding Box Extraction on Critical Hazards                                          |
|                              |                                                                   |
|                              v                                                                   |
|   Aerospace Canvas HUD Synthesis:                                                                |
|   - Crosshairs & Pitch Ladder                                                                    |
|   - Live Airspeed & Altitude Telemetry                                                           |
|   - DGCA Wind Vector & Safety Status (Safe / Caution / Grounded)                                 |
|   - Real-Time Hazard Percentage Readouts                                                         |
|                              |                                                                   |
|                              v                                                                   |
|   FastAPI WebSocket Stream: `/api/aerial/live-stream/ws`  ==>  Interactive Frontend Canvas HUD  |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulation of the Ken Burns Flight Trajectory

For an input orthomosaic of dimensions $W_0 \times H_0$, the flight simulation generates $T=60$ sequential frames ($t \in [0, T-1]$):

### 2.1. Dynamic Optical Zoom & Altitude
Simulates an aerial drone climbing from $h_0 = 80\text{ meters}$ to $h_{\max} = 120\text{ meters}$ above ground level (AGL):

$$s(t) = s_{\min} + (s_{\max} - s_{\min}) \cdot \sin^2\left(\frac{\pi t}{2(T-1)}\right), \quad s_{\min} = 0.65, \, s_{\max} = 0.90$$

$$h(t) = \frac{h_0}{s(t)} \quad (\text{meters AGL})$$

### 2.2. Smooth Panning Trajectory
Simulates a diagonal search pass across the disaster sector using cubic Hermite interpolation:

$$\Delta x(t) = \frac{W_0 - s(t) W_0}{2} \cdot \left[1 + \cos\left(\frac{\pi t}{T-1} + \phi_x\right)\right]$$

$$\Delta y(t) = \frac{H_0 - s(t) H_0}{2} \cdot \left[1 + \sin\left(\frac{\pi t}{T-1} + \phi_y\right)\right]$$

### 2.3. Simulated Aircraft Telemetry
- **Air Speed:** $v(t) = 38.0 + 4.5 \cdot \sin\left(\frac{2\pi t}{15}\right) \text{ km/h}$
- **Simulated Heading:** $\psi(t) = (\psi_0 + 0.4 \cdot t) \pmod{360^\circ}$
- **Pitch Angle:** $\theta(t) = -5.0^\circ + 2.0^\circ \cdot \sin\left(\frac{\pi t}{10}\right)$
- **Roll / Banking:** $\phi(t) = 3.5^\circ \cdot \cos\left(\frac{\pi t}{12}\right)$

---

## 3. Real-Time PyTorch Inference per Frame

Unlike static video overlays that replay hardcoded graphics, NETRA-D performs **genuine neural forward passes on each extracted frame**:
1. Frame $t$ is cropped and resized to $256 \times 256$ pixels.
2. The PyTorch tensor is normalized and passed through the segmentation model.
3. Connected components exceeding area threshold $A_{\min} = 150\text{ px}$ are extracted as active hazard bounding boxes:
   $$B_j = (x_{\min}, y_{\min}, w, h, \text{class\_id}, \text{confidence})$$
4. Dynamic bounding boxes are tagged with tactical labels:
   - 🔴 `[CRITICAL] Submerged Structure` (Red `#EF4444`)
   - 🟠 `[WARNING] Debris Flow on Road` (Orange `#F97316`)
   - 🔵 `[ALERT] Riparian Surge` (Cyan `#06B6D4`)

---

## 4. Directorate General of Civil Aviation (DGCA) Weather Engine

NETRA-D integrates live meteorological ingestion (`backend/aerial/weather.py`) that audits UAV operating conditions against official **Directorate General of Civil Aviation (DGCA)** regulations:

| Meteorological Dimension | Operational Limit | DGCA Flight Status | Visual Alert Badge | Action Directive |
| :--- | :--- | :---: | :---: | :--- |
| **Sustained Wind Speed** | $< 30\text{ km/h}$ | 🟢 **SAFE TO FLY** | `GREEN HUD` | Normal tactical reconnaissance authorized. |
| **Moderate Gusts** | $30\text{ km/h} \le v \le 45\text{ km/h}$ | 🟡 **FLIGHT CAUTION** | `AMBER HUD` | Restrict flight ceiling to $<60\text{m}$ AGL; monitor battery drain. |
| **Gale-Force Mountain Winds**| $> 45\text{ km/h}$ | 🔴 **FLIGHT GROUNDED** | `RED WARNING` | **Mandatory UAV Grounding:** Severe turbulence hazard in mountain gorge. |
| **Heavy Precipitation** | Active Rain / Cloudburst | 🔴 **FLIGHT GROUNDED** | `RED WARNING` | Ground operations immediately to protect optical payload. |

---

## 5. Ethical Transparency & Honesty Badges

To maintain scientific integrity during hackathon judging and official state disaster reviews:
- **Transparent Honesty Badge:** If an uploaded static photo is being streamed via the Ken Burns trajectory, the HUD clearly displays:
  > `[SIMULATED FLIGHT SORTIE — 60-FRAME KEN BURNS TRAJECTORY (LIVE PYTORCH INFERENCE)]`
- **Pre-Recorded Drone Video:** If actual drone video footage is uploaded, the badge reads:
  > `[PRE-RECORDED FIELD SORTIE — REAL TIME FRAME ANALYSIS]`
This ensures evaluators never mistake synthetic flight maneuvers for physical drone hardware in flight.

---

## 6. WebSocket Protocol Specification

- **Endpoint:** `WS /api/aerial/live-stream/ws`

### Server-to-Client Frame Message Schema:
```json
{
  "frame_index": 24,
  "total_frames": 60,
  "frame_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "telemetry": {
    "altitude_m": 94.2,
    "air_speed_kmh": 41.2,
    "heading_deg": 142.5,
    "pitch_deg": -4.2,
    "roll_deg": 1.8,
    "wind_speed_kmh": 22.4,
    "flight_safety": "SAFE_TO_FLY",
    "hazard_summary": {
      "flooded_pct": 14.2,
      "debris_pct": 8.6,
      "blocked_roads": 1
    },
    "bounding_boxes": [
      {
        "x": 84, "y": 120, "width": 64, "height": 48,
        "label": "Submerged Structure",
        "color": "#EF4444",
        "confidence": 0.94
      }
    ]
  }
}
```
The frontend listens to this WebSocket and renders the tactical crosshairs, telemetry gauges, and bounding boxes at 20 frames per second.
