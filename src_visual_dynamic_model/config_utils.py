from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None
    try:
        import tomli as _tomli
    except ModuleNotFoundError:
        _tomli = None


def _parse_scalar(raw: str):
    s = raw.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def load_toml_compat(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if tomllib is not None:
        return tomllib.loads(text)
    if _tomli is not None:
        return _tomli.loads(text)
    # Minimal fallback parser for simple key=value configs.
    data = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = _parse_scalar(value)
    return data

