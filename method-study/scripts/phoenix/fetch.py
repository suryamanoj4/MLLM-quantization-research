"""Build a *mini-COCO* big enough for Phoenix and nothing more.

The full COCO 2014 download is ~20 GB. Nothing here is trained, so what the project
actually needs is:

  * a few hundred evaluation images (POPE questions + CHAIR captions),
  * a couple of hundred calibration images, disjoint from those,
  * the object and caption ground truth for exactly those images.

That is ~100 MB. Two tricks get us there:

1. COCO serves individual images over HTTP
   (http://images.cocodataset.org/val2014/COCO_val2014_000000000042.jpg), so we
   fetch only the ids we selected instead of the 6.2 GB val2014 zip.

2. `annotations_trainval2014.zip` is 241 MB and expands to ~1.35 GB, but we only
   want two of its six members. This module reads the zip's central directory over
   HTTP range requests and pulls just those members (~55 MB), then subsets them to
   the selected images and throws the rest away. Falls back to a normal download if
   the server refuses ranges.

Output layout is exactly what phoenix/data.py already expects, plus a splits.json
manifest that pins the calibration/evaluation partition so the two can never leak
into each other.
"""
from __future__ import annotations

import json
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Sequence

COCO_IMAGE_BASE = "http://images.cocodataset.org"
ANNOTATIONS_ZIP = f"{COCO_IMAGE_BASE}/annotations/annotations_trainval2014.zip"
POPE_BASE = "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco"
POPE_SPLITS = ("random", "popular", "adversarial")

USER_AGENT = "phoenix-mini-coco/1.0"


# --------------------------------------------------------------------------- #
# progress bars
# --------------------------------------------------------------------------- #
try:
    from tqdm import tqdm as _tqdm
except ImportError:                                   # pragma: no cover
    _tqdm = None


def progress(total=None, desc="", unit="B", initial=0, **kw):
    """tqdm with byte units, or a minimal stderr fallback if tqdm is missing."""
    if _tqdm is not None:
        scale = unit == "B"
        return _tqdm(total=total, desc=desc, unit=unit, unit_scale=scale,
                     unit_divisor=1024 if scale else 1000, initial=initial,
                     dynamic_ncols=True, mininterval=0.3, **kw)
    return _NoTqdm(total, desc, initial)


class _NoTqdm:
    def __init__(self, total, desc, initial):
        self.total, self.desc, self.n = total, desc, initial

    def update(self, k=1):
        self.n += k
        sys.stderr.write(f"\r{self.desc} {self.n}/{self.total or '?'}")
        sys.stderr.flush()

    def set_postfix_str(self, *_a, **_k):
        pass

    def write(self, msg):
        sys.stderr.write("\n" + msg + "\n")

    def close(self):
        sys.stderr.write("\n")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# --------------------------------------------------------------------------- #
# http helpers
# --------------------------------------------------------------------------- #
# Per-socket-operation timeout. A connection that delivers nothing for this long is
# abandoned and the transfer resumes from the last byte received.
READ_TIMEOUT = 30
READ_BLOCK = 64 << 10


def _req(url: str, headers: dict | None = None, method: str = "GET"):
    h = {"User-Agent": USER_AGENT}
    h.update(headers or {})
    return urllib.request.Request(url, headers=h, method=method)


def http_get(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_req(url), timeout=timeout) as r:
                return r.read()
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


def http_range_stream(url: str, start: int, end: int,
                      timeout: int = READ_TIMEOUT) -> Iterable[bytes]:
    """Stream an inclusive byte range in small blocks.

    Requires 206 Partial Content. A 200 means the server ignored the Range header
    and is about to send the *entire* file, which must never be treated as a chunk.
    """
    req = _req(url, {"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 206:
            raise RuntimeError(f"server ignored the Range header (HTTP {r.status})")
        rd = getattr(r, "read1", r.read)
        while True:
            b = rd(READ_BLOCK)
            if not b:
                return
            yield b


def http_range(url: str, start: int, end: int, timeout: int = READ_TIMEOUT,
               retries: int = 3) -> bytes:
    """Small inclusive range read (headers, central directory)."""
    last = None
    for attempt in range(retries):
        try:
            data = b"".join(http_range_stream(url, start, end, timeout))
            if len(data) != end - start + 1:
                raise RuntimeError(f"short read: {len(data)} of {end - start + 1} bytes")
            return data
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"range GET {url} [{start}-{end}] failed: {last}")


def http_size_and_ranges(url: str, timeout: int = READ_TIMEOUT) -> tuple[int, bool]:
    with urllib.request.urlopen(_req(url, method="HEAD"), timeout=timeout) as r:
        size = int(r.headers.get("Content-Length", "0"))
        ranges = (r.headers.get("Accept-Ranges", "") or "").lower() == "bytes"
    return size, ranges


# --------------------------------------------------------------------------- #
# minimal remote-zip reader (central directory over range requests)
# --------------------------------------------------------------------------- #
EOCD_SIG = b"PK\x05\x06"
EOCD64_LOC_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CEN_SIG = b"PK\x01\x02"


class RemoteZip:
    """Read selected members of a remote zip without downloading the whole file."""

    def __init__(self, url: str, timeout: int = READ_TIMEOUT):
        self.url = url
        self.timeout = timeout
        self.size, ok = http_size_and_ranges(url, timeout)
        if not ok or self.size <= 0:
            raise RuntimeError("server does not advertise byte ranges")
        self.entries = self._read_central_directory()

    def _read_central_directory(self) -> dict[str, dict]:
        tail_len = min(self.size, 65_536 + 22)
        tail = http_range(self.url, self.size - tail_len, self.size - 1, self.timeout)
        i = tail.rfind(EOCD_SIG)
        if i < 0:
            raise RuntimeError("no end-of-central-directory record found")
        cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])

        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:        # zip64
            j = tail.rfind(EOCD64_LOC_SIG)
            if j < 0:
                raise RuntimeError("zip64 locator missing")
            (eocd64_off,) = struct.unpack("<Q", tail[j + 8:j + 16])
            head = http_range(self.url, eocd64_off, eocd64_off + 55, self.timeout)
            if not head.startswith(EOCD64_SIG):
                raise RuntimeError("bad zip64 eocd")
            cd_size, cd_off = struct.unpack("<QQ", head[40:56])

        cd = http_range(self.url, cd_off, cd_off + cd_size - 1, self.timeout)
        entries, p = {}, 0
        while p + 46 <= len(cd) and cd[p:p + 4] == CEN_SIG:
            (method, _t, _d, crc, csize, usize, nlen, elen, clen,
             _disk, _ia, _ea, lho) = struct.unpack("<HHHIIIHHHHHII", cd[p + 10:p + 46])
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
            if 0xFFFFFFFF in (csize, usize, lho):
                csize, usize, lho = _zip64_extra(extra, csize, usize, lho)
            entries[name] = {"method": method, "crc": crc, "csize": csize,
                             "usize": usize, "offset": lho}
            p += 46 + nlen + elen + clen
        if not entries:
            raise RuntimeError("central directory parsed to zero entries")
        return entries

    def namelist(self) -> list[str]:
        return sorted(self.entries)

    def _data_span(self, name: str) -> tuple[int, int]:
        e = self.entries[name]
        head = http_range(self.url, e["offset"], e["offset"] + 29, self.timeout)
        if head[:4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {name}")
        nlen, elen = struct.unpack("<HH", head[26:30])
        start = e["offset"] + 30 + nlen + elen
        return start, start + e["csize"] - 1

    def download_member(self, name: str, dest: Path, retries: int = 8,
                        desc: str | None = None) -> Path:
        """Fetch a member's *compressed* bytes to `dest`, resumably.

        Resumes from the last byte received both within a run (a stalled or dropped
        connection is retried from where it stopped, not from the start) and across
        runs (an existing partial file is continued).
        """
        e = self.entries[name]
        start, end = self._data_span(name)
        total = e["csize"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        have = dest.stat().st_size if dest.exists() else 0
        if have > total:
            dest.unlink()
            have = 0

        label = desc or Path(name).name
        with progress(total=total, initial=have, desc=f"[ann] {label}") as bar, \
                open(dest, "ab") as f:
            failures = 0
            while have < total:
                try:
                    for blk in http_range_stream(self.url, start + have, end, self.timeout):
                        f.write(blk)
                        have += len(blk)
                        bar.update(len(blk))
                        failures = 0
                    f.flush()
                except Exception as ex:               # noqa: BLE001
                    failures += 1
                    if failures > retries:
                        raise RuntimeError(
                            f"{name}: gave up after {retries} consecutive failures "
                            f"at {have/1e6:.1f}/{total/1e6:.1f} MB ({ex}). "
                            "Rerun to resume from here.") from None
                    wait = min(30, 2 ** failures)
                    bar.write(f"[ann] connection problem at {have/1e6:.1f} MB ({ex}); "
                              f"resuming in {wait}s (attempt {failures}/{retries})")
                    time.sleep(wait)
        return dest

    def download_member_parallel(self, name: str, dest: Path, connections: int = 8,
                                 deadline: float | None = None, retries: int = 8,
                                 desc: str | None = None, bar=None) -> dict:
        """Fetch a member's compressed bytes over `connections` parallel ranges.

        Each connection owns one contiguous segment written to its own file, so both
        in-run retries and cross-run resumes continue from the exact byte reached.
        With `deadline` (a time.monotonic() value) it stops early and reports the
        throughput it saw -- that is how the auto mode measures the range path.
        Returns {"done", "bytes", "seconds"}.
        """
        import threading
        e = self.entries[name]
        start, end = self._data_span(name)
        total = e["csize"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.stat().st_size == total:
                return {"done": True, "bytes": 0, "seconds": 0.0}
            dest.unlink()                 # partial single-stream file from an older run

        n = max(1, min(connections, total // (256 << 10) or 1))
        bounds = [(total * i // n, total * (i + 1) // n) for i in range(n)]
        segs = [dest.with_name(f"{dest.name}.seg{i}of{n}") for i in range(n)]
        for other in dest.parent.glob(f"{dest.name}.seg*"):
            if other not in segs:         # segment layout changed; start clean
                other.unlink()
        have = [sg.stat().st_size if sg.exists() else 0 for sg in segs]
        lock = threading.Lock()
        errors: list[BaseException] = []
        got = [0]
        t0 = time.monotonic()

        own_bar = bar is None
        if own_bar:
            bar = progress(total=total, initial=sum(have),
                           desc=f"[ann] {desc or Path(name).name} x{n}")

        def worker(i: int):
            lo, hi = bounds[i]
            failures = 0
            with open(segs[i], "ab") as f:
                while have[i] < hi - lo:
                    if deadline is not None and time.monotonic() > deadline:
                        return
                    try:
                        for blk in http_range_stream(self.url, start + lo + have[i],
                                                     start + hi - 1, self.timeout):
                            f.write(blk)
                            with lock:
                                have[i] += len(blk)
                                got[0] += len(blk)
                                bar.update(len(blk))
                            failures = 0
                            if deadline is not None and time.monotonic() > deadline:
                                return
                    except Exception as ex:           # noqa: BLE001
                        failures += 1
                        if failures > retries:
                            errors.append(ex)
                            return
                        time.sleep(min(20, 2 ** failures))

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        secs = time.monotonic() - t0
        done = all(have[i] >= bounds[i][1] - bounds[i][0] for i in range(n))
        if own_bar:
            bar.close()
        if errors and not done and deadline is None:
            raise RuntimeError(f"{name}: a segment failed {retries} times in a row "
                               f"({errors[0]}); rerun to resume") from None
        if done:
            with open(dest, "wb") as out:
                for sg in segs:
                    with open(sg, "rb") as fi:
                        while True:
                            b = fi.read(8 << 20)
                            if not b:
                                break
                            out.write(b)
            for sg in segs:
                sg.unlink(missing_ok=True)
        return {"done": done, "bytes": got[0], "seconds": secs}

    def open_member(self, name: str) -> Iterable[bytes]:
        """Yield decompressed bytes of one member, streaming (no resume)."""
        e = self.entries[name]
        start, end = self._data_span(name)
        dec = zlib.decompressobj(-15) if e["method"] == 8 else None
        for raw in http_range_stream(self.url, start, end, self.timeout):
            yield dec.decompress(raw) if dec else raw
        if dec:
            tail = dec.flush()
            if tail:
                yield tail


def probe_get_speed(url: str, seconds: float = 6.0) -> float:
    """Bytes/s of a plain (non-range) GET over `seconds`; the data is discarded."""
    t0 = time.monotonic()
    n = 0
    try:
        with urllib.request.urlopen(_req(url), timeout=READ_TIMEOUT) as r:
            rd = getattr(r, "read1", r.read)
            while time.monotonic() - t0 < seconds:
                b = rd(READ_BLOCK)
                if not b:
                    break
                n += len(b)
    except Exception:                                 # noqa: BLE001
        return 0.0
    return n / max(time.monotonic() - t0, 1e-6)


def inflate_member(src: Path, dest: Path, method: int, usize: int, crc: int,
                   desc: str = "") -> Path:
    """Decompress a raw zip member body and verify its size and CRC-32."""
    dec = zlib.decompressobj(-15) if method == 8 else None
    got_crc, n = 0, 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(src, "rb") as fi, open(tmp, "wb") as fo, \
            progress(total=usize, desc=f"[ann] inflate {desc}") as bar:
        while True:
            raw = fi.read(4 << 20)
            if not raw:
                break
            out = dec.decompress(raw) if dec else raw
            fo.write(out)
            got_crc = zlib.crc32(out, got_crc)
            n += len(out)
            bar.update(len(out))
        if dec:
            out = dec.flush()
            fo.write(out)
            got_crc = zlib.crc32(out, got_crc)
            n += len(out)
            bar.update(len(out))
    if n != usize or (got_crc & 0xFFFFFFFF) != crc:
        tmp.unlink(missing_ok=True)
        src.unlink(missing_ok=True)
        raise RuntimeError(f"{desc}: integrity check failed (size {n} vs {usize}, "
                           f"crc {got_crc & 0xFFFFFFFF:08x} vs {crc:08x}); the partial "
                           "download was deleted, rerun to fetch it again")
    tmp.rename(dest)
    return dest


def _zip64_extra(extra: bytes, csize: int, usize: int, lho: int):
    p = 0
    while p + 4 <= len(extra):
        hid, hsz = struct.unpack("<HH", extra[p:p + 4])
        if hid == 0x0001:
            vals = list(struct.unpack(f"<{hsz // 8}Q", extra[p + 4:p + 4 + (hsz // 8) * 8]))
            it = iter(vals)
            if usize == 0xFFFFFFFF:
                usize = next(it, usize)
            if csize == 0xFFFFFFFF:
                csize = next(it, csize)
            if lho == 0xFFFFFFFF:
                lho = next(it, lho)
            break
        p += 4 + hsz
    return csize, usize, lho


# --------------------------------------------------------------------------- #
# annotation acquisition
# --------------------------------------------------------------------------- #
WANTED_MEMBERS = {
    "instances": "annotations/instances_{subset}.json",
    "captions": "annotations/captions_{subset}.json",
}


def _fmt_rate(bps: float) -> str:
    return f"{bps/1e6:.2f} MB/s" if bps >= 1e5 else f"{bps/1e3:.0f} kB/s"


def _fmt_eta(sec: float) -> str:
    if sec == float("inf"):
        return "never"
    return f"{sec/60:.1f} min" if sec >= 90 else f"{sec:.0f} s"


def fetch_annotation_members(subset: str, want: Sequence[str],
                             cache_dir: Path, url: str = ANNOTATIONS_ZIP,
                             log: Callable[[str], None] = print,
                             mode: str = "auto", connections: int = 8,
                             probe_seconds: float = 8.0) -> dict[str, Path]:
    """Return {kind: path to the full annotation json}, as fast as the link allows.

    mode="range": pull only the needed members via parallel HTTP range requests
                  (~55 MB transferred).
    mode="full":  one plain GET of the whole zip (~253 MB), extract, delete.
    mode="auto":  measure both for a few seconds and take whichever finishes first.
                  Some networks (campus proxies; CDNs forwarding uncached ranges to
                  origin) serve range requests orders of magnitude slower than a
                  plain GET, so "less data" is not always "less time".
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    todo = []
    for kind in want:
        member = WANTED_MEMBERS[kind].format(subset=subset)
        dest = cache_dir / Path(member).name
        if dest.exists() and dest.stat().st_size > 0:
            log(f"[ann] cached {dest.name} ({dest.stat().st_size/1e6:.0f} MB)")
            out[kind] = dest
        else:
            todo.append((kind, member, dest))
    if not todo:
        return out

    zpath = cache_dir / "annotations_trainval2014.zip"
    rz = None
    if mode != "full" and not zpath.exists():
        try:
            rz = RemoteZip(url)
        except Exception as e:                        # noqa: BLE001
            log(f"[ann] range requests unavailable ({e}); using the full zip")

    if rz is not None:
        for _, member, _ in todo:
            if member not in rz.entries:
                raise RuntimeError(f"{member} not in zip; members: {rz.namelist()[:8]}")
        need = sum(rz.entries[m]["csize"] for _, m, _ in todo)
        log(f"[ann] zip is {rz.size/1e6:.0f} MB; the {len(todo)} needed member(s) are "
            f"{need/1e6:.0f} MB compressed")

        if mode == "auto":
            kind0, member0, dest0 = todo[0]
            raw0 = dest0.with_suffix(".deflate.part")
            log(f"[ann] probing range speed ({connections} connections, "
                f"{probe_seconds:.0f} s) ...")
            try:
                r = rz.download_member_parallel(
                    member0, raw0, connections=connections,
                    deadline=time.monotonic() + probe_seconds, desc="probe")
            except Exception as ex:                   # noqa: BLE001
                r = {"done": False, "bytes": 0, "seconds": probe_seconds}
                log(f"[ann] range probe failed ({ex})")
            if not r["done"]:
                range_bps = r["bytes"] / max(r["seconds"], 1e-6)
                left = need - r["bytes"] - _partial_bytes(raw0)
                log(f"[ann] probing plain-GET speed ({probe_seconds*0.75:.0f} s) ...")
                full_bps = probe_get_speed(url, probe_seconds * 0.75)
                t_range = left / range_bps if range_bps > 0 else float("inf")
                t_full = rz.size / full_bps if full_bps > 0 else float("inf")
                log(f"[ann]   ranges : {_fmt_rate(range_bps):>11}  -> "
                    f"~{_fmt_eta(t_range)} for the remaining {left/1e6:.0f} MB")
                log(f"[ann]   plain  : {_fmt_rate(full_bps):>11}  -> "
                    f"~{_fmt_eta(t_full)} for the whole {rz.size/1e6:.0f} MB zip")
                if t_full < 0.75 * t_range:
                    log("[ann] -> plain GET wins; downloading the full zip once, "
                        "extracting 2 files, then deleting it")
                    for sg in cache_dir.glob("*.seg*of*"):
                        sg.unlink(missing_ok=True)
                    rz = None
                else:
                    log("[ann] -> ranges win; continuing")

    if rz is not None:
        for kind, member, dest in todo:
            e = rz.entries[member]
            raw = dest.with_suffix(".deflate.part")
            try:
                rz.download_member_parallel(member, raw, connections=connections)
            except Exception as ex:                   # noqa: BLE001
                raise explain_network_error(ex, url) from None
            inflate_member(raw, dest, e["method"], e["usize"], e["crc"], dest.name)
            raw.unlink(missing_ok=True)
            out[kind] = dest
        return out

    # ---- full-zip path ----------------------------------------------------- #
    import zipfile
    if not zpath.exists():
        # restart rather than resume: this path is taken precisely when range
        # requests (which a resume would need) are the slow thing on this network
        zpath.with_suffix(zpath.suffix + ".part").unlink(missing_ok=True)
        log(f"[ann] downloading {url}")
        try:
            _download(url, zpath)
        except Exception as e:                        # noqa: BLE001
            raise explain_network_error(e, url) from None
    with zipfile.ZipFile(zpath) as z:
        for kind, member, dest in todo:
            info = z.getinfo(member)
            with z.open(member) as src, open(dest, "wb") as f, \
                    progress(total=info.file_size, desc=f"[ann] extract {dest.name}") as bar:
                while True:
                    b = src.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    bar.update(len(b))
            out[kind] = dest
    zpath.unlink(missing_ok=True)
    log("[ann] removed the full zip (only the 2 extracted files are kept until subsetting)")
    return out


def _partial_bytes(raw: Path) -> int:
    return sum(p.stat().st_size for p in raw.parent.glob(f"{raw.name}.seg*"))


class FetchError(RuntimeError):
    """A network failure with an actionable message attached."""


def explain_network_error(e: BaseException, url: str) -> FetchError:
    host = re.sub(r"^https?://([^/]+).*$", r"\1", url)
    code = getattr(e, "code", None)
    lines = [f"could not fetch {url}", f"  {type(e).__name__}: {e}", ""]
    if code in (403, 407) or "Forbidden" in str(e) or "Tunnel" in str(e):
        lines += [
            f"  {host} refused the request. Usually one of:",
            "    * a corporate/university proxy or firewall blocks it "
            "(set HTTP_PROXY / HTTPS_PROXY),",
            "    * the host is reachable only over plain HTTP and your network "
            "rewrites it,",
            "    * you are on a sandboxed machine with an egress allowlist.",
        ]
    else:
        lines += [f"  {host} did not answer. Check connectivity and retry; the "
                  "script resumes from what is already downloaded."]
    lines += [
        "",
        "  Workarounds:",
        "    * download annotations_trainval2014.zip by hand (241 MB) and pass",
        "      --annotations-zip /path/to/annotations_trainval2014.zip",
        "    * or point --root at an existing full COCO checkout; everything else "
        "works unchanged.",
    ]
    return FetchError("\n".join(lines))


def _download(url: str, dest: Path, timeout: int = READ_TIMEOUT,
              retries: int = 8) -> None:
    """Whole-file download with a progress bar, resuming via Range when possible."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        total, ranges = http_size_and_ranges(url, timeout)
    except Exception:                                 # noqa: BLE001
        total, ranges = 0, False
    have = tmp.stat().st_size if (tmp.exists() and ranges) else 0
    failures = 0
    with progress(total=total or None, initial=have, desc=f"[dl] {dest.name}") as bar:
        while True:
            try:
                hdr = {"Range": f"bytes={have}-"} if (ranges and have) else {}
                with urllib.request.urlopen(_req(url, hdr), timeout=timeout) as r, \
                        open(tmp, "ab" if hdr else "wb") as f:
                    rd = getattr(r, "read1", r.read)
                    while True:
                        blk = rd(READ_BLOCK)
                        if not blk:
                            break
                        f.write(blk)
                        have += len(blk)
                        bar.update(len(blk))
                        failures = 0
                if not total or have >= total:
                    break
            except Exception as ex:                   # noqa: BLE001
                failures += 1
                if failures > retries or not ranges:
                    raise
                wait = min(30, 2 ** failures)
                bar.write(f"[dl] connection problem at {have/1e6:.1f} MB ({ex}); "
                          f"resuming in {wait}s")
                time.sleep(wait)
    tmp.rename(dest)


# --------------------------------------------------------------------------- #
# subsetting
# --------------------------------------------------------------------------- #
def subset_instances(src: Path, keep_ids: set[int], dest: Path,
                     log: Callable[[str], None] = print) -> dict[int, set[str]]:
    data = _load_coco_json(src, keep_ids, log)
    cats = {c["id"]: c["name"] for c in data["categories"]}
    objs: dict[int, set[str]] = {}
    for a in data["annotations"]:
        objs.setdefault(a["image_id"], set()).add(cats[a["category_id"]])
    _write_json(dest, data)
    log(f"[ann] wrote {dest.name}: {len(data['images'])} images, "
        f"{len(data['annotations'])} annotations, {dest.stat().st_size/1e6:.1f} MB")
    return objs


def subset_captions(src: Path, keep_ids: set[int], dest: Path,
                    log: Callable[[str], None] = print) -> None:
    data = _load_coco_json(src, keep_ids, log, need_categories=False)
    _write_json(dest, data)
    log(f"[ann] wrote {dest.name}: {len(data['annotations'])} captions, "
        f"{dest.stat().st_size/1e6:.1f} MB")


def _write_json(dest: Path, data) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(data, f)


class _ProgressReader:
    """File wrapper that advances a byte progress bar as ijson reads it."""

    def __init__(self, f, bar):
        self.f, self.bar = f, bar

    def read(self, n=-1):
        b = self.f.read(n)
        self.bar.update(len(b))
        return b


def _ijson_fast():
    """Return ijson only if it has a C backend; the pure-Python one parses at a few
    MB/s, which turns a 161 MB file into several silent minutes -- json.load is
    faster there, at the cost of ~1.5 GB of RAM for a moment."""
    try:
        import ijson                                  # noqa: PLC0415
    except ImportError:
        return None
    backend = getattr(ijson, "backend", "")
    if "yajl2" in backend:
        return ijson
    try:
        return ijson.get_backend("yajl2_c")
    except Exception:                                 # noqa: BLE001
        return None


# segmentation polygons are most of the bytes and nothing here reads them
_DROP_FIELDS = ("segmentation",)


def _slim(a: dict) -> dict:
    return {k: v for k, v in a.items() if k not in _DROP_FIELDS}


def _load_coco_json(src: Path, keep_ids: set[int],
                    log: Callable[[str], None] = print,
                    need_categories: bool = True) -> dict:
    """Load and filter a COCO annotation file, streaming when a fast ijson exists."""
    size = src.stat().st_size
    ijson = _ijson_fast()
    if ijson is None:
        log(f"[ann] parsing {src.name} in memory ({size/1e6:.0f} MB, ~10-30 s; "
            "`pip install ijson` to stream it instead)")
        with open(src) as f:
            data = json.load(f)
        return {
            "info": data.get("info", {}), "licenses": data.get("licenses", []),
            "categories": data.get("categories", []) if need_categories else [],
            "images": [im for im in data["images"] if im["id"] in keep_ids],
            "annotations": [_slim(a) for a in data["annotations"]
                            if a["image_id"] in keep_ids],
        }

    out = {"info": {}, "licenses": [], "categories": [], "images": [], "annotations": []}
    passes = [("images.item", "images", lambda x: x["id"] in keep_ids, lambda x: x),
              ("annotations.item", "annotations", lambda x: x["image_id"] in keep_ids,
               _slim)]
    if need_categories:
        passes.append(("categories.item", "categories", lambda x: True, lambda x: x))
    for k, (prefix, key, keep, shape) in enumerate(passes, 1):
        with open(src, "rb") as f, progress(
                total=size, desc=f"[ann] scan {src.name} {k}/{len(passes)} ({key})") as bar:
            for item in ijson.items(_ProgressReader(f, bar), prefix):
                if keep(item):
                    out[key].append(shape(_plain(item)))
    return out


def _plain(o):
    """ijson yields Decimal for numbers; make it JSON-serialisable again."""
    from decimal import Decimal
    if isinstance(o, Decimal):
        f = float(o)
        return int(f) if f.is_integer() else f
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_plain(v) for v in o]
    return o


# --------------------------------------------------------------------------- #
# images
# --------------------------------------------------------------------------- #
def image_url(image_id: int, subset: str = "val2014") -> str:
    if subset.endswith("2014"):
        return f"{COCO_IMAGE_BASE}/{subset}/COCO_{subset}_{image_id:012d}.jpg"
    return f"{COCO_IMAGE_BASE}/{subset}/{image_id:012d}.jpg"


def image_dest(root: Path, image_id: int, subset: str = "val2014") -> Path:
    if subset.endswith("2014"):
        return root / subset / f"COCO_{subset}_{image_id:012d}.jpg"
    return root / subset / f"{image_id:012d}.jpg"


def download_images(root: Path, image_ids: Sequence[int], subset: str = "val2014",
                    workers: int = 16, log: Callable[[str], None] = print) -> list[int]:
    """Fetch the given images. Returns the ids that failed."""
    (root / subset).mkdir(parents=True, exist_ok=True)
    todo = [i for i in image_ids
            if not (image_dest(root, i, subset).exists()
                    and image_dest(root, i, subset).stat().st_size > 1024)]
    if not todo:
        log(f"[img] all {len(image_ids)} {subset} images already present")
        return []
    log(f"[img] fetching {len(todo)} images into {root/subset} "
        f"({len(image_ids)-len(todo)} cached)")
    try:                                              # fail fast and legibly
        http_get(image_url(todo[0], subset), timeout=60, retries=2)
    except Exception as e:                            # noqa: BLE001
        raise explain_network_error(e, image_url(todo[0], subset)) from None

    failed = []

    def one(iid: int):
        dest = image_dest(root, iid, subset)
        try:
            data = http_get(image_url(iid, subset), timeout=READ_TIMEOUT, retries=3)
            if len(data) < 1024:
                raise RuntimeError("suspiciously small response")
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(dest)
            return iid, len(data)
        except Exception:                             # noqa: BLE001
            return iid, None

    nbytes = 0
    with ThreadPoolExecutor(max_workers=workers) as ex, \
            progress(total=len(todo), desc="[img] images", unit="img") as bar:
        # as_completed, not map: map yields in submission order, so one slow image
        # would freeze the bar while the other 15 workers keep finishing
        for fut in as_completed([ex.submit(one, i) for i in todo]):
            iid, n = fut.result()
            if n is None:
                failed.append(iid)
            else:
                nbytes += n
            bar.update(1)
            bar.set_postfix_str(f"{nbytes/1e6:.1f} MB, {len(failed)} failed")
    return failed


def bytes_on_disk(root: Path) -> int:
    return sum(f.stat().st_size for f in Path(root).rglob("*") if f.is_file())


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n} B"


# --------------------------------------------------------------------------- #
# POPE
# --------------------------------------------------------------------------- #
def fetch_pope(dest_dir: Path, splits: Sequence[str] = POPE_SPLITS,
               log: Callable[[str], None] = print) -> dict[str, Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in splits:
        p = dest_dir / f"coco_pope_{s}.json"
        if not p.exists() or p.stat().st_size == 0:
            u = f"{POPE_BASE}/coco_pope_{s}.json"
            try:
                p.write_bytes(http_get(u))
            except Exception as e:                    # noqa: BLE001
                raise explain_network_error(e, u) from None
            log(f"[pope] fetched {p.name} ({p.stat().st_size/1e3:.0f} KB)")
        out[s] = p
    return out


def pope_image_ids(path: Path) -> list[int]:
    ids, seen = [], set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(r"(\d{6,})", json.loads(line)["image"])
        if not m:
            continue
        iid = int(m.group(1))
        if iid not in seen:
            seen.add(iid)
            ids.append(iid)
    return ids


def filter_pope(src: Path, keep_ids: set[int], dest: Path) -> int:
    kept = []
    for line in Path(src).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        m = re.search(r"(\d{6,})", d["image"])
        if m and int(m.group(1)) in keep_ids:
            kept.append(json.dumps(d))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(kept) + ("\n" if kept else ""))
    return len(kept)


# --------------------------------------------------------------------------- #
# split manifest
# --------------------------------------------------------------------------- #
SPLITS_FILE = "splits.json"


def write_splits(root: Path, eval_ids: Sequence[int], calib_ids: Sequence[int],
                 subset: str, meta: dict | None = None) -> Path:
    assert not (set(eval_ids) & set(calib_ids)), "eval/calib overlap -- refusing to write"
    p = Path(root) / SPLITS_FILE
    payload = {"subset": subset, "eval_ids": sorted(eval_ids),
               "calib_ids": sorted(calib_ids), "meta": meta or {}}
    p.write_text(json.dumps(payload, indent=2))
    return p


def read_splits(root: str | Path) -> dict | None:
    p = Path(root) / SPLITS_FILE
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["eval_ids"] = [int(i) for i in d["eval_ids"]]
    d["calib_ids"] = [int(i) for i in d["calib_ids"]]
    overlap = set(d["eval_ids"]) & set(d["calib_ids"])
    if overlap:
        raise RuntimeError(f"{p} has {len(overlap)} images in both splits")
    return d
