"""Normalisation helpers: numbers, fuel/hybrid/transmission/body canonical values, feature tags."""
from __future__ import annotations

import re
from datetime import datetime

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

CANON_FUELS = ("Petrol", "Diesel", "CNG", "Electric", "Hybrid")


def num(s) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = _NUM_RE.search(str(s))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def to_int(s) -> int | None:
    v = num(s)
    return int(round(v)) if v is not None else None


def positive(v):
    return v if v is not None and v > 0 else None


def lakh_text_to_inr(text: str | None) -> tuple[int | None, int | None]:
    """'Rs. 20.00 - 33.00 Lakh' / 'Rs. 1.20 Crore' / 'Rs. 95 Lakh - 1.2 Crore' -> (min, max) in INR."""
    if not text:
        return None, None
    t = text.replace(",", "")
    parts = re.findall(r"(\d+(?:\.\d+)?)\s*(Lakh|Crore|Cr|L)?", t, flags=re.I)
    vals, last_unit = [], None
    for value, unit in reversed(parts):  # units usually trail the range: "20 - 33 Lakh"
        unit = (unit or last_unit or "Lakh").lower()
        last_unit = unit
        mult = 1e7 if unit.startswith("c") else 1e5
        vals.append(int(round(float(value) * mult)))
    vals = [v for v in reversed(vals) if v >= 1e5]
    if not vals:
        return None, None
    return min(vals), max(vals)


def parse_date(s: str | None) -> str | None:
    """CarWale '12/28/2022 00:00:00' -> '2022-12-28'."""
    if not s:
        return None
    for fmt in ("%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def canon_fuel(raw: str | None) -> str | None:
    if not raw:
        return None
    r = raw.lower()
    if "hybrid" in r and "mild" not in r:
        return "Hybrid"
    if "electric" in r and "petrol" not in r and "diesel" not in r:
        return "Electric"
    if "cng" in r:
        return "CNG"
    if "diesel" in r:
        return "Diesel"
    if "petrol" in r or "mild" in r:
        return "Petrol"
    if "electric" in r:
        return "Electric"
    return raw.strip().title()


_MILD_CUES = ("mild hybrid", "smart hybrid", "shvs", "48v", "48 v", "isg", "integrated starter generator", "mhev")
_STRONG_CUES = ("strong hybrid", "self-charging", "self charging", "e:hev", "ehev", "e-cvt", "intelligent electric hybrid")


def hybrid_type(fuel_raw: str | None, *texts: str | None) -> str | None:
    """Classify a powertrain as 'strong', 'mild', 'plug-in' or None from fuel label and engine/spec text."""
    f = (fuel_raw or "").lower()
    blob = " ".join(t.lower() for t in texts if t)
    if "plug-in" in f or "plug in" in f or "phev" in blob or "plug-in hybrid" in blob:
        return "plug-in"
    if "mild" in f or any(c in blob for c in _MILD_CUES):
        return "mild"
    if "hybrid" in f or any(c in blob for c in _STRONG_CUES):
        return "strong"
    return None


def canon_transmission(raw: str | None) -> tuple[str | None, bool | None]:
    if not raw:
        return None, None
    r = raw.lower()
    if "imt" in r or "clutchless" in r:
        return "iMT", False
    if "manual" in r and "automatic" not in r:
        return "Manual", False
    if "e-cvt" in r or "ecvt" in r:
        return "e-CVT", True
    if "cvt" in r:
        return "CVT", True
    if "amt" in r:
        return "AMT", True
    if "dct" in r or "dsg" in r or "dual clutch" in r:
        return "DCT", True
    if "automatic" in r or "tc" in r or "torque" in r:
        return "AT" if ("tc" in r or "torque" in r) else "Automatic", True
    return raw.strip(), None


def canon_body(raw: str | None) -> str | None:
    """Collapse CarWale body styles into a small filterable set."""
    if not raw:
        return None
    r = raw.lower()
    if "suv" in r:
        return "SUV"
    if "muv" in r or "mpv" in r:
        return "MUV"
    if "hatch" in r:
        return "Hatchback"
    if "sedan" in r:
        return "Sedan"
    if "coupe" in r:
        return "Coupe"
    if "convertible" in r or "cabriolet" in r or "roadster" in r:
        return "Convertible"
    if "pick" in r or "truck" in r:
        return "Pickup"
    if "van" in r:
        return "Van"
    if "wagon" in r or "estate" in r:
        return "Wagon"
    return raw.strip()


def parse_ncap(s: str | None) -> tuple[int | None, str | None]:
    """'5 Star (Bharat NCAP)' -> (5, 'Bharat NCAP')."""
    if not s:
        return None, None
    m = re.search(r"(\d)\s*Star", s, flags=re.I)
    if not m:
        return None, None
    agency = re.search(r"\(([^)]*NCAP[^)]*)\)", s, flags=re.I)
    return int(m.group(1)), (agency.group(1).strip() if agency else None)


def yes(v) -> bool:
    if v is None:
        return False
    s = str(v).strip().lower()
    return bool(s) and s not in ("no", "0", "none", "not applicable", "na", "n/a", "-")


FEATURE_TAGS = [
    "sunroof", "panoramic_sunroof", "ventilated_seats", "wireless_charger", "360_camera", "reverse_camera",
    "cruise_control", "adaptive_cruise", "adas", "blind_spot_monitor", "connected_car", "ota_updates",
    "apple_carplay_android_auto", "wireless_carplay", "auto_climate", "dual_zone_climate", "hud",
    "tpms", "esc", "hill_hold", "6_airbags", "4x4", "ambient_lighting", "leather_seats", "powered_seats",
    "led_headlights", "keyless_start", "rear_ac_vents", "turbo", "dashcam", "front_parking_sensors",
    "touchscreen", "aeb", "lane_keep_assist", "voice_commands", "electric_parking_brake",
]


def variant_tags(specs: dict[str, str], airbags: int | None) -> list[str]:
    """Derive filterable feature tags from a variant's spec dict (itemName -> value)."""
    g = lambda k: specs.get(k, "")  # noqa: E731
    tags = set()
    sunroof = g("Sunroof Type")
    if yes(sunroof):
        tags.add("sunroof")
        if "panoramic" in sunroof.lower():
            tags.add("panoramic_sunroof")
    if yes(g("Ventilated Seats")):
        tags.add("ventilated_seats")
    if yes(g("Wireless Charger")):
        tags.add("wireless_charger")
    pa = g("Parking Assist").lower()
    if "360" in pa:
        tags.update({"360_camera", "reverse_camera"})
    elif "camera" in pa:
        tags.add("reverse_camera")
    cc = g("Cruise Control")
    if yes(cc):
        tags.add("cruise_control")
        if "adaptive" in cc.lower():
            tags.update({"adaptive_cruise", "adas"})
    if yes(g("ADAS")) or yes(g("Automatic Emergency Braking")) or yes(g("Automatic Emergency Braking ")):
        tags.add("adas")
    if yes(g("Blind Spot Detection")):
        tags.add("blind_spot_monitor")
    if yes(g("Apple CarPlay")) or yes(g("Android Auto")):
        tags.add("apple_carplay_android_auto")
        if "wireless" in (g("Apple CarPlay") + g("Android Auto")).lower():
            tags.add("wireless_carplay")
    ac = g("Air Conditioner Control").lower()
    if "automatic" in ac or "climate" in ac:
        tags.add("auto_climate")
        if "dual" in ac or "zone" in ac:
            tags.add("dual_zone_climate")
    if yes(g("Heads Up Display (HUD)")):
        tags.add("hud")
    if yes(g("Tyre Pressure Monitoring System")):
        tags.add("tpms")
    if yes(g("Electronic Stability Program (ESP)")):
        tags.add("esc")
    if yes(g("Hill Hold Control")):
        tags.add("hill_hold")
    if airbags and airbags >= 6:
        tags.add("6_airbags")
    if yes(g("Four-Wheel-Drive")) or g("Drivetrain").upper() in ("4WD", "AWD", "4X4"):
        tags.add("4x4")
    if yes(g("Ambient Interior Lighting")):
        tags.add("ambient_lighting")
    if "leather" in g("Seat Upholstery").lower():
        tags.add("leather_seats")
    if "electric" in g("Seat: Forward / Back").lower():
        tags.add("powered_seats")
    if "led" in g("Headlights").lower():
        tags.add("led_headlights")
    if yes(g("Keyless Start/ Button Start")):
        tags.add("keyless_start")
    if yes(g("Rear AC Vents")):
        tags.add("rear_ac_vents")
    if yes(g("Turbocharger / Supercharger")):
        tags.add("turbo")
    if yes(g("Dashcam")):
        tags.add("dashcam")
    if "front" in g("Parking Sensors").lower():
        tags.add("front_parking_sensors")
    if "touch" in g("Infotainment (Front)").lower() or yes(g("Touch-screen Display")):
        tags.add("touchscreen")
    if yes(g("Voice Command")):
        tags.add("voice_commands")
    return sorted(tags)


def tree_tags(groups: dict[str, list[dict]]) -> list[str]:
    """Model-level tags from the top trim's full feature tree ({group: [{name, value}]})."""
    tags = set()
    app_yes = 0
    for group, items in groups.items():
        gl = group.lower()
        for it in items:
            name, val = it["name"].lower(), it["value"]
            if not yes(val):
                continue
            if "mobile app" in gl or "remote" in gl or "connected" in name:
                app_yes += 1
            if "over the air" in name or "ota" in name.split():
                tags.add("ota_updates")
            if name.strip() == "adas":
                tags.add("adas")
            if "automatic emergency braking" in name:
                tags.update({"aeb", "adas"})
            if "lane" in name and ("keep" in str(val).lower() or "keep" in name):
                tags.add("lane_keep_assist")
            if "electronic parking brake" in name or "electric parking brake" in name:
                tags.add("electric_parking_brake")
    if app_yes >= 2:
        tags.add("connected_car")
    return sorted(tags)
