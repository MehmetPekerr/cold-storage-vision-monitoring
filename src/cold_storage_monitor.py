"""
Depo Kapı ve Zemin Takip Modülü
-------------------------------
Depodaki kapının açık/kapalı durumunu (debounce filtreli) ve
koridor zemin doluluk durumunu (YOLO + dinamik referans farkı) takip eder.
Kapı açık kalma süresi 60 sn'yi veya zemin doluluğu 1 saati aşarsa alarm üretir.

Config example:
    {
        "type":                      "depo_takip",
        "name":                      "cold_storage_camera",
        "kapi_model_path":           "door_model.pt",
        "zemin_model_path":          "floor_model.pt",
        "ref_kapali_path":           "bos_koridor_kapali.jpg",
        "ref_acik_path":             "bos_koridor_acik.jpg",
        "door_polygon":              [[311, 0], [308, 145], [392, 151], [404, 0]],
        "corridor_polygon":          [[316, 142], [249, 573], [451, 575], [382, 148]],
        "door_open_timeout_sec":     60.0,
        "door_debounce_sec":         5.0,
        "floor_empty_tolerance_sec": 20.0,
        "floor_alert_sec":           3600.0,
        "show_panel":                true
    }

get_data() output:
    {
        "Door State":             str,    # "acik" | "kapali"
        "Total Open Seconds":     float,
        "Total Open Minutes":     float,
        "Door Open Count":        int,
        "Floor Occupied":         bool,
        "Floor Occupancy Pct":    float,
        "Floor Occupied Minutes": float,
        "door_open_alert":        bool,
        "floor_occupied_alert":   bool
    }
"""

# pyright: reportMissingImports=false
# pyright: reportGeneralTypeIssues=false
# pylint: disable=import-error,no-name-in-module
# flake8: noqa

import os
import json
import time as _time
import datetime
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Tür B modül gereksinimleri (GPU Lock + Model)
try:
    from engine.shared_memory import GPU_LOCK  # type: ignore
except (ImportError, ModuleNotFoundError):
    try:
        from core.shared_memory import GPU_LOCK  # type: ignore
    except (ImportError, ModuleNotFoundError):
        import threading
        GPU_LOCK = threading.Lock()

try:
    from .base import BaseModule  # type: ignore
except (ImportError, ValueError):
    try:
        from base import BaseModule  # type: ignore
    except ImportError:
        class BaseModule:  # type: ignore
            def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
                raise NotImplementedError

            def get_data(self) -> dict:
                raise NotImplementedError

            def draw(self, frame):
                return frame

            def reset(self):
                pass

            def shutdown(self):
                pass


# ==================== CONFIG ====================

STATE_DIR         = "state"
SAVE_INTERVAL_SEC = 30

# Varsayılan Geometri ve Eşik Değerleri
DEFAULT_DOOR_PTS = [
    [311, 0],
    [308, 145],
    [392, 151],
    [404, 0]
]

DEFAULT_CORRIDOR_PTS = [
    [316, 142],
    [249, 573],
    [451, 575],
    [382, 148]
]

COLOR_CLOSED = (0, 255, 0)      # Yeşil
COLOR_OPEN   = (0, 255, 255)    # Sarı
COLOR_ALERT  = (0, 0, 255)      # Kırmızı
COLOR_FLOOR  = (255, 0, 0)      # Mavi


# ==================== MODULE ====================

class DepoTakipModule(BaseModule):

    def __init__(self, name: str,
                 kapi_model_path: str = "door_model.pt",
                 zemin_model_path: str = "floor_model.pt",
                 ref_kapali_path: str = "bos_koridor_kapali.jpg",
                 ref_acik_path: str = "bos_koridor_acik.jpg",
                 door_polygon: list = None,
                 corridor_polygon: list = None,
                 door_open_timeout_sec: float = 60.0,
                 door_debounce_sec: float = 5.0,
                 floor_empty_tolerance_sec: float = 20.0,
                 floor_alert_sec: float = 3600.0,
                 show_panel: bool = True,
                 **_kwargs):
        self.name                      = name
        self.kapi_model_path           = kapi_model_path
        self.zemin_model_path          = zemin_model_path
        self.ref_kapali_path           = ref_kapali_path
        self.ref_acik_path             = ref_acik_path
        self.door_open_timeout_sec     = door_open_timeout_sec
        self.door_debounce_sec         = door_debounce_sec
        self.floor_empty_tolerance_sec = floor_empty_tolerance_sec
        self.floor_alert_sec           = floor_alert_sec
        self.show_panel                = show_panel

        # Poligon Geometrileri
        door_list = door_polygon if door_polygon is not None else DEFAULT_DOOR_PTS
        corridor_list = corridor_polygon if corridor_polygon is not None else DEFAULT_CORRIDOR_PTS
        self._door_pts = np.array(door_list, np.int32)
        self._corridor_pts = np.array(corridor_list, np.int32)
        self._dx, self._dy, self._dw, self._dh = cv2.boundingRect(self._door_pts)
        self._cx, self._cy, self._cw, self._ch = cv2.boundingRect(self._corridor_pts)

        # Tür B Modellerini Yükle
        self._kapi_model = None
        self._zemin_model = None
        if YOLO is not None:
            if os.path.exists(self.kapi_model_path):
                try:
                    self._kapi_model = YOLO(self.kapi_model_path)
                except Exception as e:
                    print(f"[{self.name}] Kapi model load error: {e}")
            else:
                print(f"[{self.name}] Kapi model file not found: {self.kapi_model_path}")

            if os.path.exists(self.zemin_model_path):
                try:
                    self._zemin_model = YOLO(self.zemin_model_path)
                except Exception as e:
                    print(f"[{self.name}] Zemin model load error: {e}")

        # Referans Kareleri (Zemin Fark Analizi)
        self._ref_frame_kapali = None
        self._ref_frame_acik = None

        # Kapı Durum Değişkenleri
        self._door_yolo_state = "kapali"
        self._door_state_change_time = _time.time()
        self._door_logical_state = "kapali"
        self._door_logical_open_start = None
        self._door_open_count = 0
        self._total_open_seconds = 0.0
        self._door_open_alert = False

        # Zemin Durum Değişkenleri
        self._floor_is_occupied = False
        self._floor_occupied_start_time = None
        self._floor_last_seen_occupied = None
        self._floor_occupied_seconds = 0.0
        self._floor_occupancy_pct = 0.0
        self._floor_alert = False

        # Kalıcılık ve Günlük Sıfırlama
        self._last_reset_date = None
        self._last_save_time = 0.0

        # draw() için durum saklayıcı (update çağrılmadan önce None)
        self._last_status = None

        self._load_state()
        print(f"✅ DepoTakipModule ready [{name}]")

    # ==================== PERSISTENCE ====================

    def _state_path(self) -> str:
        return os.path.join(STATE_DIR, f"depo_takip_{self.name}.json")

    def _save_state(self):
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            state = {
                "date": (self._last_reset_date.isoformat()
                         if self._last_reset_date else None),
                "total_open_seconds":     self._total_open_seconds,
                "door_open_count":        self._door_open_count,
                "floor_occupied_seconds": self._floor_occupied_seconds,
            }
            tmp = self._state_path() + f".{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            for attempt in range(5):
                try:
                    os.replace(tmp, self._state_path())
                    break
                except OSError:
                    if attempt < 4:
                        _time.sleep(0.05)
                    else:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                        raise
        except Exception as e:
            print(f"[{self.name}] State save error: {e}")

    def _load_state(self):
        path = self._state_path()
        if not os.path.exists(path):
            print(f"[{self.name}] No state file, starting fresh.")
            return
        try:
            with open(path, encoding="utf-8") as f:
                state = json.load(f)
            saved_date = state.get("date")
            today = datetime.datetime.now().date().isoformat()
            if saved_date != today:
                print(f"[{self.name}] State outdated ({saved_date}), starting fresh.")
                return
            self._total_open_seconds     = float(state.get("total_open_seconds", 0.0))
            self._door_open_count        = int(state.get("door_open_count", 0))
            self._floor_occupied_seconds = float(state.get("floor_occupied_seconds", 0.0))
            self._last_reset_date        = datetime.date.fromisoformat(saved_date)
            print(f"[{self.name}] State loaded (Open: {self._total_open_seconds/60:.1f} min, Count: {self._door_open_count})")
        except Exception as e:
            print(f"[{self.name}] State load error: {e} — starting fresh.")

    # ==================== HELPERS ====================

    def _check_daily_reset(self):
        today = datetime.datetime.now().date()
        if self._last_reset_date is None:
            self._last_reset_date = today
            return
        if today != self._last_reset_date:
            self._total_open_seconds     = 0.0
            self._door_open_count        = 0
            self._floor_occupied_seconds = 0.0
            now = _time.time()
            if self._door_logical_state == "acik":
                self._door_logical_open_start = now
            if self._floor_is_occupied:
                self._floor_occupied_start_time = now
            self._last_reset_date = today
            self._save_state()
            print(f"[{self.name}] Daily reset → {today}")

    def _prepare_ref_image(self, filename: str, target_h: int, target_w: int):
        if not os.path.exists(filename):
            return None
        try:
            bgr = cv2.imread(filename)
            if bgr is None:
                return None
            bgr = cv2.resize(bgr, (target_w, target_h))
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # Simetrik ayna düzeltmesi (koridor temiz zemin referansı)
            mid_x, start_x, end_x = 350, 249, 451
            if end_x <= target_w and start_x >= 0 and mid_x <= end_x:
                gray[:, mid_x:end_x] = cv2.flip(gray[:, start_x:mid_x], 1)
            return cv2.GaussianBlur(gray, (21, 21), 0)
        except Exception as e:
            print(f"[{self.name}] Ref image error ({filename}): {e}")
            return None

    # ==================== UPDATE ====================

    def update(self, bboxes, class_ids, scores, object_ids, frame, class_names: dict):
        self._check_daily_reset()
        now = _time.time()
        h, w = frame.shape[:2]

        # Referans kareleri ilk kare geldiğinde hazırla
        if self._ref_frame_kapali is None:
            self._ref_frame_kapali = self._prepare_ref_image(self.ref_kapali_path, h, w)
            self._ref_frame_acik   = self._prepare_ref_image(self.ref_acik_path, h, w)

        # ---------------------------------------------------------
        # 1. KAPI ANALİZİ (YOLO + 5 SN DEBOUNCE FİLTRESİ)
        # ---------------------------------------------------------
        if self._kapi_model is not None and self._dw > 0 and self._dh > 0:
            door_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(door_mask, [self._door_pts], 255)
            isolated_door = cv2.bitwise_and(frame, frame, mask=door_mask)
            cropped_door = isolated_door[self._dy:self._dy+self._dh, self._dx:self._dx+self._dw]

            # Tür B: GPU_LOCK altında tek inference
            if torch is not None:
                with GPU_LOCK, torch.no_grad():
                    results = self._kapi_model(cropped_door, verbose=False)
            else:
                with GPU_LOCK:
                    results = self._kapi_model(cropped_door, verbose=False)

            # Lock dışında işle
            predicted_class = results[0].names[results[0].probs.top1]
            current_yolo = "acik" if predicted_class == "kapi_acik" else "kapali"

            if current_yolo != self._door_yolo_state:
                self._door_yolo_state = current_yolo
                self._door_state_change_time = now

            time_since_change = now - self._door_state_change_time

            # 5 saniyelik onay filtresi
            if self._door_logical_state == "kapali":
                if self._door_yolo_state == "acik" and time_since_change >= self.door_debounce_sec:
                    self._door_logical_state = "acik"
                    self._door_logical_open_start = now
                    self._door_open_count += 1
            elif self._door_logical_state == "acik":
                if self._door_yolo_state == "kapali" and time_since_change >= self.door_debounce_sec:
                    self._door_logical_state = "kapali"
                    if self._door_logical_open_start:
                        duration = now - self._door_logical_open_start
                        self._total_open_seconds += max(0.0, duration)
                        self._door_logical_open_start = None
                    self._door_open_alert = False

            # Kapı 60 sn açık kalma alarmı
            if self._door_logical_state == "acik" and self._door_logical_open_start:
                if (now - self._door_logical_open_start) > self.door_open_timeout_sec:
                    self._door_open_alert = True
                else:
                    self._door_open_alert = False
            else:
                self._door_open_alert = False

        # ---------------------------------------------------------
        # 2. ZEMİN ANALİZİ (YOLO + DOLULUK YÜZDESİ)
        # ---------------------------------------------------------
        detected_floor_boxes = []
        if self._zemin_model is not None and self._cw > 0 and self._ch > 0:
            floor_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(floor_mask, [self._corridor_pts], 255)
            isolated_floor = cv2.bitwise_and(frame, frame, mask=floor_mask)
            cropped_floor = isolated_floor[self._cy:self._cy+self._ch, self._cx:self._cx+self._cw]

            # Tür B: GPU_LOCK altında tek inference
            if torch is not None:
                with GPU_LOCK, torch.no_grad():
                    floor_results = self._zemin_model(cropped_floor, verbose=False)
            else:
                with GPU_LOCK:
                    floor_results = self._zemin_model(cropped_floor, verbose=False)

            # Lock dışında işle
            floor_class_id = floor_results[0].probs.top1
            floor_class_name = floor_results[0].names[floor_class_id]

            if floor_class_name == "dolu":
                self._floor_last_seen_occupied = now
                if not self._floor_is_occupied:
                    self._floor_is_occupied = True
                    self._floor_occupied_start_time = now

                # Doluluk Yüzdesi Hesaplama (Fark Analizi)
                active_ref = (self._ref_frame_acik
                              if (self._door_logical_state == "acik" and self._ref_frame_acik is not None)
                              else self._ref_frame_kapali)

                if active_ref is not None:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
                    frame_diff = cv2.absdiff(active_ref, gray_frame)
                    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
                    corridor_thresh = cv2.bitwise_and(thresh, floor_mask)
                    contours, _ = cv2.findContours(corridor_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    boxes_mask = np.zeros_like(thresh)
                    for c in contours:
                        if cv2.contourArea(c) > 300:
                            bx, by, bw, bh = cv2.boundingRect(c)
                            detected_floor_boxes.append((bx, by, bw, bh))
                            cv2.rectangle(boxes_mask, (bx, by), (bx + bw, by + bh), 255, -1)

                    intersection = cv2.bitwise_and(boxes_mask, floor_mask)
                    occupied_pixels = cv2.countNonZero(intersection)
                    total_floor_pixels = cv2.countNonZero(floor_mask)
                    if total_floor_pixels > 0:
                        self._floor_occupancy_pct = (occupied_pixels / total_floor_pixels) * 100.0
                    else:
                        self._floor_occupancy_pct = 0.0
                else:
                    self._floor_occupancy_pct = 50.0  # Referans yoksa gösterge
            else:
                # 20 saniye tolerans (geçici boşluklarda sıfırlanmaz)
                if self._floor_is_occupied:
                    if self._floor_last_seen_occupied and (now - self._floor_last_seen_occupied) > self.floor_empty_tolerance_sec:
                        self._floor_is_occupied = False
                        if self._floor_occupied_start_time:
                            self._floor_occupied_seconds += (now - self._floor_occupied_start_time)
                            self._floor_occupied_start_time = None
                        self._floor_occupancy_pct = 0.0
                        self._floor_alert = False

            # Zemin sürekli 1 saat dolu kalırsa alarm
            if self._floor_is_occupied and self._floor_occupied_start_time:
                elapsed_floor = now - self._floor_occupied_start_time
                if elapsed_floor >= self.floor_alert_sec:
                    self._floor_alert = True
                else:
                    self._floor_alert = False
            else:
                self._floor_alert = False

        # draw() için son durumu sakla
        self._last_status = {
            "door_logical_state": self._door_logical_state,
            "door_open_alert":    self._door_open_alert,
            "floor_is_occupied":  self._floor_is_occupied,
            "floor_occupancy_pct":self._floor_occupancy_pct,
            "floor_alert":        self._floor_alert,
            "floor_boxes":        detected_floor_boxes,
        }

        # Periyodik State Kaydı (SAVE_INTERVAL_SEC)
        if now - self._last_save_time >= SAVE_INTERVAL_SEC:
            self._save_state()
            self._last_save_time = now

    # ==================== DATA ====================

    def get_data(self) -> dict:
        now = _time.time()
        total_open = self._total_open_seconds
        if self._door_logical_state == "acik" and self._door_logical_open_start:
            total_open += now - self._door_logical_open_start

        floor_total = self._floor_occupied_seconds
        if self._floor_is_occupied and self._floor_occupied_start_time:
            floor_total += now - self._floor_occupied_start_time

        return {
            "Door State":             self._door_logical_state,
            "Total Open Seconds":     round(total_open, 1),
            "Total Open Minutes":     round(total_open / 60.0, 1),
            "Door Open Count":        self._door_open_count,
            "Floor Occupied":         self._floor_is_occupied,
            "Floor Occupancy Pct":    round(self._floor_occupancy_pct, 1),
            "Floor Occupied Minutes": round(floor_total / 60.0, 1),
            "door_open_alert":        self._door_open_alert,
            "floor_occupied_alert":   self._floor_alert,
        }

    # ==================== DRAW ====================

    def draw(self, frame):
        if self._last_status is None:
            return frame

        # 1. Kapı Alanı ve Durumu
        door_state = self._last_status["door_logical_state"]
        door_color = COLOR_OPEN if door_state == "acik" else COLOR_CLOSED
        door_text = "Open" if door_state == "acik" else "Closed"
        cv2.polylines(frame, [self._door_pts], isClosed=True, color=door_color, thickness=2)
        cv2.putText(frame, door_text, (self._dx + self._dw + 10, self._dy + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, door_color, 2, cv2.LINE_AA)

        # 2. Zemin / Koridor Alanı ve Doluluğu
        cv2.polylines(frame, [self._corridor_pts], isClosed=True, color=COLOR_FLOOR, thickness=2)

        # Dolu olan zemin nesnelerini kırmızı kutu ile çiz
        if self._last_status.get("floor_is_occupied"):
            for bx, by, bw, bh in self._last_status.get("floor_boxes", []):
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_ALERT, 2)

        pct = self._last_status["floor_occupancy_pct"]
        floor_text_x = 250
        floor_text_y = self._cy + 45
        cv2.putText(frame, f"%{pct:.0f}", (floor_text_x, floor_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_FLOOR, 2, cv2.LINE_AA)
        cv2.putText(frame, "Occupied", (floor_text_x, floor_text_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_FLOOR, 2, cv2.LINE_AA)

        # 3. Bilgi Paneli ve Alarmlar (show_panel=True ise)
        if self.show_panel:
            frame = self._draw_panel(frame)

        return frame

    def _draw_panel(self, frame):
        now_dt = datetime.datetime.now()
        time_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, time_str, (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ALERT, 2, cv2.LINE_AA)

        # Kapı 60 sn açık alarmı
        if self._last_status.get("door_open_alert"):
            cv2.putText(frame, "WARNING: DOOR OPEN FOR OVER 60 SECONDS!", (15, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ALERT, 2, cv2.LINE_AA)

        # Zemin 1 saat dolu alarmı
        if self._last_status.get("floor_alert"):
            cv2.putText(frame, "WARNING: FLOOR CONTINUOUSLY OCCUPIED > 1 HOUR!", (15, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ALERT, 2, cv2.LINE_AA)

        return frame

    # ==================== SHUTDOWN ====================

    def shutdown(self):
        now = _time.time()
        if self._door_logical_state == "acik" and self._door_logical_open_start:
            self._total_open_seconds += max(0.0, now - self._door_logical_open_start)
            self._door_logical_open_start = None
            self._door_logical_state = "kapali"

        if self._floor_is_occupied and self._floor_occupied_start_time:
            self._floor_occupied_seconds += max(0.0, now - self._floor_occupied_start_time)
            self._floor_occupied_start_time = None
            self._floor_is_occupied = False

        self._save_state()
        print(f"[{self.name}] Shutdown complete, state saved.")

    def reset(self):
        self._total_open_seconds = 0.0
        self._door_open_count = 0
        self._floor_occupied_seconds = 0.0
        self._door_logical_state = "kapali"
        self._door_logical_open_start = None
        self._floor_is_occupied = False
        self._floor_occupied_start_time = None
        self._save_state()
