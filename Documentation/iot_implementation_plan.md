# IoT Integration Implementation Plan

## 1. Goal Description
To integrate an IoT (Internet of Things) component into the software-only GEO-SRM project without requiring physical hardware (Arduino/Raspberry Pi). We will use a free mobile app (like **Blynk IoT**) to act as a simulated ground sensor. This will allow the software to fuse "Macro" satellite data with "Micro" ground-level IoT data for real-time validation.

## 2. Proposed Changes

### Component 1: Mobile App Setup (No Code)
*   **Action:** You will download the **Blynk IoT** app from the Play Store.
*   **Setup:** Create a simple project in the app with a "Slider" or "Virtual Sensor" (e.g., simulating Soil Moisture or Temperature).
*   **API Key:** Blynk will provide an Auth Token and a REST API URL to read the value of your phone's simulated sensor over the internet.

---

### Component 2: Backend (FastAPI Integration)

#### [NEW] `k:/PROJECTS/try-1/backend/api/iot_routes.py`
*   Create a new FastAPI router dedicated to IoT.
*   Implement a `GET /api/iot-sensor` endpoint.
*   This endpoint will make an HTTP request to the Blynk Cloud API using your Auth Token to fetch the live value you set on your phone.

#### [MODIFY] `k:/PROJECTS/try-1/backend/app.py`
*   Import the new `iot_routes` router.
*   Include the router in the main FastAPI application (`app.include_router(iot_routes)`).

---

### Component 3: Frontend (Dashboard UI)

#### [MODIFY] `k:/PROJECTS/try-1/frontend/index.html`
*   Add a new "IoT Sensor Data" floating widget or panel on the dashboard.
*   Add a button: "Fetch Live Ground Truth (IoT)".

#### [MODIFY] `k:/PROJECTS/try-1/frontend/app.js`
*   Add an event listener to the new IoT button.
*   When clicked, it will make a fetch request to `/api/iot-sensor`.
*   Update the dashboard widget dynamically with the data coming straight from your phone in real-time.

---

## 3. User Review Required
> [!IMPORTANT]
> **Blynk Setup:** You will need to install the Blynk app on your phone and generate an Auth Token. Are you comfortable doing this quick 5-minute setup on your end, so I can write the exact code for it?

> [!NOTE]
> **Use Case Selection:** Do you want the simulated IoT sensor to represent **Soil Moisture** (for agriculture), **Water Level** (for flood detection), or **Temperature**? I will tailor the UI text accordingly.

## 4. Verification Plan
*   **Manual Testing:** We will change the slider on your mobile phone, click "Fetch" on the web dashboard, and verify that the exact value appears on the screen in real-time.
