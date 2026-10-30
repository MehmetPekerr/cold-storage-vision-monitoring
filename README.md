# Cold Storage Vision Monitoring

Real-time computer-vision monitoring for cold-storage environments. The system combines door-state classification, corridor occupancy analysis, temporal filtering, configurable alerts, persistence, and live video overlays.

> **Public demo and data policy:** This repository contains no company footage, camera URLs, reports, model weights, or private metadata. The included demo creates cartoon-style synthetic frames locally. Production deployments should use approved, access-controlled data and separately supplied model artifacts.

## What the original system does

- Classifies a cold-storage door as open or closed with a custom YOLO classifier.
- Uses debounce filtering to prevent transient predictions from becoming events.
- Tracks door-open events, duration, daily totals, and configurable timeout alerts.
- Classifies a corridor as empty or occupied with a second YOLO model.
- Selects an open-door or closed-door reference frame before calculating occupancy.
- Estimates occupied floor percentage using polygon regions of interest and image differences.
- Draws status overlays, regions, counters, and alert indicators on live frames.
- Persists daily state in JSON and resets counters at the start of a new day.
- Supports integration with an existing camera-processing pipeline through `BaseModule`.

The public demo intentionally uses a lightweight synthetic detector instead of the private production models. This keeps the repository runnable without exposing proprietary training data or weights.

## Repository layout

```text
.
├── demo/
│   └── generate_synthetic_demo.py   Creates cartoon-style frames and an annotated demo video
├── src/
│   ├── base.py                      Pipeline module interface
│   └── cold_storage_monitor.py      Production-oriented monitoring module
├── data/
│   └── synthetic/                   Generated locally; ignored by Git
├── .gitignore
├── requirements.txt
└── README.md
```

## Run the safe synthetic demo

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
python demo/generate_synthetic_demo.py
```

The demo writes generated frames and `synthetic_monitor_demo.mp4` under `data/synthetic/`. Three small synthetic preview frames are included; the generated video remains excluded from version control.

## Production integration

The production-oriented module expects model artifacts to be supplied outside this repository:

```python
from src.cold_storage_monitor import DepoTakipModule

monitor = DepoTakipModule(
    name="cold_storage_camera",
    kapi_model_path="path/to/approved/door_model.pt",
    zemin_model_path="path/to/approved/floor_model.pt",
    ref_kapali_path="path/to/approved/empty_closed.jpg",
    ref_acik_path="path/to/approved/empty_open.jpg",
)
```

Do not commit camera credentials, RTSP URLs, real frames, customer reports, state files, or model weights. Load deployment-specific values from an environment variable or a private configuration store.

## Limitations

The synthetic demo is for reproducibility and architecture demonstration only. It is not a benchmark and must not be used to claim production accuracy. Real-world performance depends on camera placement, lighting, domain-specific training data, calibration, and operational validation.

## License and data rights

Only add a license after confirming that the source code and any included assets can be redistributed. This public package includes synthetic assets only; no customer or company data is distributed.
