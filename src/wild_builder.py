from __future__ import annotations
import argparse
import hashlib
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import requests
from src.utils.config import load_yaml, ensure_dir
from src.utils.seed import seed_everything

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"

def _safe_get(d: Dict[str, Any], path: List[str], default: str = "") -> str:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return str(cur) if cur is not None else default

def _extmetadata_field(extmeta: Dict[str, Any], key: str) -> str:
    if key not in extmeta:
        return ""
    v = extmeta[key]
    if isinstance(v, dict):
        return str(v.get("value", "")).strip()
    return str(v).strip()

@dataclass
class WikiImage:
    class_name: str
    pageid: int
    title: str
    url: str
    width: int
    height: int
    mime: str
    license_short: str
    license_url: str
    artist: str
    credit: str
    attribution_required: str
    usage_terms: str
    source: str

def wikimedia_search_files(
    session: requests.Session,
    query: str,
    limit: int,
    sleep_s: float,
    user_agent: str,
) -> List[Dict[str, Any]]:
    """
    1. list=search to get File: titles in namespace 6
    2. query titles with prop=imageinfo to get URLs + extmetadata
    """
    headers = {"User-Agent": user_agent}

    titles: List[str] = []
    sroffset = 0

    while len(titles) < limit:
        params_search = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srnamespace": 6,
            "srsearch": query,
            "srlimit": min(50, limit - len(titles)),
            "sroffset": sroffset,
        }
        r = session.get(WIKIMEDIA_API, params=params_search, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            print("Wikimedia API error:", data["error"])

        batch = data.get("query", {}).get("search", [])
        if not batch:
            break

        for item in batch:
            t = item.get("title")
            if t:
                titles.append(t)

        cont = data.get("continue", {})
        if "sroffset" not in cont:
            break
        sroffset = cont["sroffset"]

        time.sleep(sleep_s)

    titles = titles[:limit]
    print(f"[wikimedia] query='{query}' titles_found={len(titles)}")

    if not titles:
        return []

    #fetch imageinfo for the titles in chunks
    pages: List[Dict[str, Any]] = []
    chunk_size = 50
    for i in range(0, len(titles), chunk_size):
        chunk = titles[i : i + chunk_size]
        params_info = {
            "action": "query",
            "format": "json",
            "titles": "|".join(chunk),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
        }
        r = session.get(WIKIMEDIA_API, params=params_info, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        pages_dict = data.get("query", {}).get("pages", {})
        pages.extend(list(pages_dict.values()))
        time.sleep(sleep_s)

    print(f"[wikimedia] imageinfo_pages={len(pages)}")
    return pages

def parse_wiki_pages_to_images(
    class_name: str,
    pages: List[Dict[str, Any]],
    min_w: int,
    min_h: int,
) -> List[WikiImage]:
    imgs: List[WikiImage] = []
    for p in pages:
        pageid = int(p.get("pageid", -1))
        title = str(p.get("title", ""))
        ii = p.get("imageinfo", [])
        if not ii:
            continue
        ii0 = ii[0]

        url = ii0.get("url", "")
        width = int(ii0.get("width", 0) or 0)
        height = int(ii0.get("height", 0) or 0)
        mime = str(ii0.get("mime", ""))

        if not url:
            continue

        if width < min_w or height < min_h:
            continue

        extmeta = ii0.get("extmetadata", {}) or {}

        lic_short = _extmetadata_field(extmeta, "LicenseShortName")
        lic_url = _extmetadata_field(extmeta, "LicenseUrl")
        artist = _extmetadata_field(extmeta, "Artist")
        credit = _extmetadata_field(extmeta, "Credit")
        attrib_req = _extmetadata_field(extmeta, "AttributionRequired")
        usage_terms = _extmetadata_field(extmeta, "UsageTerms")
        source = _extmetadata_field(extmeta, "ImageDescription")

        imgs.append(
            WikiImage(
                class_name=class_name,
                pageid=pageid,
                title=title,
                url=url,
                width=width,
                height=height,
                mime=mime,
                license_short=lic_short,
                license_url=lic_url,
                artist=artist,
                credit=credit,
                attribution_required=attrib_req,
                usage_terms=usage_terms,
                source=source,
            )
        )
    return imgs

def _file_ext_from_url(url: str) -> str:
    p = url.split("?")[0]
    ext = Path(p).suffix.lower()
    if ext and len(ext) <= 5:
        return ext
    return ".jpg"

def download_image(session: requests.Session, url: str, out_path: Path, sleep_s: float, user_agent: str) -> bool:
    headers = {"User-Agent": user_agent}
    try:
        r = session.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        time.sleep(sleep_s)
        return True
    except Exception:
        return False

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/wild.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)["wild"]

    seed = int(cfg["seed"])
    seed_everything(seed, deterministic=True)

    classes: List[str] = cfg["classes"]
    images_per_class = int(cfg["images_per_class"])
    max_candidates = int(cfg["max_candidates_per_class"])
    min_w = int(cfg["min_width"])
    min_h = int(cfg["min_height"])
    out_dir = ensure_dir(cfg["output_dir"])
    out_csv = Path(cfg["output_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    sleep_s = float(cfg.get("request_sleep_sec", 0.2))
    user_agent = str(cfg.get("user_agent", "food101-wild-builder/1.0"))

    session = requests.Session()

    all_rows: List[Dict[str, Any]] = []

    for cls in classes:
        #encourage food context in search
        q = f'{cls.replace("_", " ")} food'
        pages = wikimedia_search_files(
            session=session,
            query=q,
            limit=max_candidates,
            sleep_s=sleep_s,
            user_agent=user_agent,
        )
        imgs = parse_wiki_pages_to_images(cls, pages, min_w=min_w, min_h=min_h)

        print(f"{cls}: parsed_images_after_filters={len(imgs)}")
        if len(imgs) > 0:
            print(" example url:", imgs[0].url)
            print(" example license_short:", imgs[0].license_short)

        #deduplicate by URL
        uniq = {}
        for im in imgs:
            uniq[im.url] = im
        imgs = list(uniq.values())

        #sample deterministically
        rng = random.Random(seed + hash(cls) % 10_000)
        rng.shuffle(imgs)
        chosen = imgs[:images_per_class]

        class_dir = ensure_dir(out_dir / cls)
        print(f"{cls}: candidates={len(imgs)} chosen={len(chosen)}")

        for im in chosen:
            #stable filename: pageid + hash
            h = hashlib.sha1(im.url.encode("utf-8")).hexdigest()[:10]
            ext = _file_ext_from_url(im.url)
            fname = f"{im.pageid}_{h}{ext}"
            fpath = class_dir / fname

            ok = True
            if not fpath.exists():
                ok = download_image(session, im.url, fpath, sleep_s=sleep_s, user_agent=user_agent)

            all_rows.append(
                {
                    "class_name": im.class_name,
                    "pageid": im.pageid,
                    "title": im.title,
                    "url": im.url,
                    "local_path": str(fpath).replace("\\", "/"),
                    "width": im.width,
                    "height": im.height,
                    "mime": im.mime,
                    "license_short": im.license_short,
                    "license_url": im.license_url,
                    "artist": im.artist,
                    "credit": im.credit,
                    "attribution_required": im.attribution_required,
                    "usage_terms": im.usage_terms,
                    "download_ok": ok,
                }
            )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote metadata CSV: {out_csv}")
    print(f"Images stored under: {out_dir}")

    ok = df["download_ok"].sum()
    total = len(df)
    print(f"Download success: {ok}/{total} ({ok/total:.1%})")

if __name__ == "__main__":
    main()
