"""
release_assets.py — helper GitHub Releases (pure-Python urllib, tanpa `gh`/requests).

Dipakai untuk siklus "dataset = aset Release" (lihat docs/DATA_RELEASE_WORKFLOW.md):
  • Sisi maintenance (Colab): create_or_get_release + upload_asset → unggah tarball
    multi-part + MANIFEST hasil pack_dataset_release.py. delete_release → hapus versi lama.
  • Sisi training (Colab): pull_dataset → unduh MANIFEST, unduh semua part, verifikasi
    sha256, satukan ulang, extract ke data_dir. Idempotent.

Token: Personal Access Token (repo scope). Di Colab: userdata.get('GITHUB_TOKEN').

Catatan teknis:
  • Aset repo privat diunduh via endpoint API `/releases/assets/{id}` dgn header
    Accept: application/octet-stream → GitHub redirect 302 ke S3 (URL bertanda-tangan).
    Header Authorization HARUS DILEPAS saat redirect lintas-host, kalau tidak S3 menolak
    ("only one auth mechanism allowed"). Ditangani _NoAuthRedirect di bawah.
  • Upload/Download streaming (file besar multi-GB tidak dimuat ke memori).

CLI ringkas:
  python release_assets.py upload   --repo OWNER/REPO --tag data-v8 --files dist/*  --token $T
  python release_assets.py pull     --repo OWNER/REPO --tag data-v8 --data_dir ../3DCNN/dataset --token $T
  python release_assets.py delete   --repo OWNER/REPO --tag data-v8 --token $T
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.github.com"
UPLOADS = "https://uploads.github.com"
_UA = "thesis-release-assets/1.0"


class _NoAuthRedirect(urllib.request.HTTPRedirectHandler):
    """Lepas header Authorization saat redirect ke host berbeda (mis. S3)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old_host = urllib.parse.urlsplit(req.full_url).netloc
            new_host = urllib.parse.urlsplit(newurl).netloc
            if old_host != new_host:
                new.headers = {k: v for k, v in new.header_items()
                               if k.lower() != "authorization"}
        return new


_OPENER = urllib.request.build_opener(_NoAuthRedirect)


def _req(method, url, token, data=None, headers=None, timeout=120):
    h = {"User-Agent": _UA, "Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    if headers:
        h.update(headers)
    r = urllib.request.Request(url, data=data, headers=h, method=method)
    return _OPENER.open(r, timeout=timeout)


def _json(method, url, token, payload=None, timeout=120, retries=3):
    body = json.dumps(payload).encode() if payload is not None else None
    hdr = {"Content-Type": "application/json"} if body else None
    last = None
    for attempt in range(1, retries + 1):
        try:
            with _req(method, url, token, data=body, headers=hdr, timeout=timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        if attempt < retries:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{method} {url} gagal: {last}")


def _split_repo(repo: str):
    if "/" not in repo:
        sys.exit(f"--repo harus OWNER/REPO, dapat '{repo}'")
    owner, name = repo.split("/", 1)
    return owner, name


def _sha256(path: Path, buf: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


# ───────────────────────── release lifecycle ─────────────────────────

def get_release(repo: str, tag: str, token: str):
    owner, name = _split_repo(repo)
    try:
        return _json("GET", f"{API}/repos/{owner}/{name}/releases/tags/{tag}", token)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def create_or_get_release(repo: str, tag: str, token: str, name: str = None,
                          body: str = "", target: str = None, prerelease: bool = False):
    rel = get_release(repo, tag, token)
    if rel:
        print(f"[release] '{tag}' sudah ada (id={rel['id']}).")
        return rel
    owner, rname = _split_repo(repo)
    payload = {"tag_name": tag, "name": name or tag, "body": body, "prerelease": prerelease}
    if target:
        payload["target_commitish"] = target
    rel = _json("POST", f"{API}/repos/{owner}/{rname}/releases", token, payload)
    print(f"[release] dibuat '{tag}' (id={rel['id']}).")
    return rel


def delete_release(repo: str, tag: str, token: str, delete_tag: bool = True):
    rel = get_release(repo, tag, token)
    if not rel:
        print(f"[release] '{tag}' tidak ada — tak ada yg dihapus.")
        return
    owner, name = _split_repo(repo)
    _json("DELETE", f"{API}/repos/{owner}/{name}/releases/{rel['id']}", token)
    print(f"[release] release '{tag}' dihapus.")
    if delete_tag:
        try:
            _json("DELETE", f"{API}/repos/{owner}/{name}/git/refs/tags/{tag}", token)
            print(f"[release] git tag '{tag}' dihapus.")
        except urllib.error.HTTPError as e:
            print(f"[release] tag '{tag}' tak terhapus (HTTP {e.code}) — abaikan.")


# ───────────────────────── asset upload/download ─────────────────────────

def _delete_asset_if_exists(repo, rel, asset_name, token):
    owner, name = _split_repo(repo)
    for a in rel.get("assets", []):
        if a["name"] == asset_name:
            _json("DELETE", f"{API}/repos/{owner}/{name}/releases/assets/{a['id']}", token)
            print(f"  (replace) aset lama '{asset_name}' dihapus.")


def upload_asset(repo: str, tag: str, filepath, token: str, retries: int = 3):
    """Unggah satu file sbg aset Release '{tag}' (timpa bila nama sama)."""
    owner, name = _split_repo(repo)
    rel = create_or_get_release(repo, tag, token)
    fp = Path(filepath)
    asset_name = fp.name
    _delete_asset_if_exists(repo, rel, asset_name, token)
    size = fp.stat().st_size
    url = f"{UPLOADS}/repos/{owner}/{name}/releases/{rel['id']}/assets?name={urllib.parse.quote(asset_name)}"
    last = None
    for attempt in range(1, retries + 1):
        try:
            with open(fp, "rb") as f:
                hdr = {"Content-Type": "application/octet-stream",
                       "Content-Length": str(size)}
                # file object sbg data → urllib stream (tak dimuat ke memori)
                with _req("POST", url, token, data=f, headers=hdr, timeout=3600) as resp:
                    a = json.loads(resp.read())
                    print(f"  uploaded {asset_name} ({size/1e6:.1f}MB) → id={a['id']}")
                    return a
        except Exception as e:  # noqa: BLE001
            last = str(e)
            print(f"  upload gagal (attempt {attempt}/{retries}): {last[:200]}")
            # refresh release (asset parsial mungkin tertinggal) lalu coba lagi
            rel = create_or_get_release(repo, tag, token)
            _delete_asset_if_exists(repo, rel, asset_name, token)
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"upload {asset_name} gagal: {last}")


def upload_assets(repo: str, tag: str, files, token: str):
    for f in files:
        upload_asset(repo, tag, f, token)


def download_asset(repo: str, tag: str, asset_name: str, dest, token: str, retries: int = 3):
    """Unduh satu aset (stream ke file). Repo privat: via endpoint API + octet-stream."""
    owner, name = _split_repo(repo)
    rel = get_release(repo, tag, token)
    if not rel:
        raise RuntimeError(f"release '{tag}' tidak ada.")
    asset = next((a for a in rel.get("assets", []) if a["name"] == asset_name), None)
    if not asset:
        raise RuntimeError(f"aset '{asset_name}' tidak ada di release '{tag}'.")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{API}/repos/{owner}/{name}/releases/assets/{asset['id']}"
    last = None
    for attempt in range(1, retries + 1):
        try:
            with _req("GET", url, token, headers={"Accept": "application/octet-stream"},
                      timeout=3600) as resp:
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(8 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                os.replace(tmp, dest)
            print(f"  downloaded {asset_name} → {dest}")
            return dest
        except Exception as e:  # noqa: BLE001
            last = str(e)
            print(f"  download gagal (attempt {attempt}/{retries}): {last[:200]}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"download {asset_name} gagal: {last}")


# ───────────────────────── convenience: pull dataset ─────────────────────────

def pull_dataset(repo: str, tag: str, data_dir, token: str, workdir=None):
    """Unduh MANIFEST + semua part, verifikasi sha256, satukan, extract ke data_dir.

    Idempotent: bila data_dir sudah berisi jumlah file >= manifest.file_count → skip.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    work = Path(workdir) if workdir else data_dir.parent / "_release_dl"
    work.mkdir(parents=True, exist_ok=True)

    man_name = f"MANIFEST_{tag.replace('data-', '')}.json"
    man_path = work / man_name
    try:
        download_asset(repo, tag, man_name, man_path, token)
    except RuntimeError:
        # fallback: cari manifest apa pun di release
        rel = get_release(repo, tag, token) or {}
        cand = next((a["name"] for a in rel.get("assets", []) if a["name"].startswith("MANIFEST_")), None)
        if not cand:
            raise RuntimeError(f"MANIFEST tidak ditemukan di release '{tag}'.")
        man_name = cand
        download_asset(repo, tag, man_name, man_path, token)
    manifest = json.loads(man_path.read_text())

    # idempotent skip
    have = sum(1 for _ in data_dir.glob("*/*/frame_*/*"))
    if have >= manifest["file_count"]:
        print(f"[pull] data_dir sudah lengkap ({have} >= {manifest['file_count']}) — skip.")
        return manifest

    # unduh + verifikasi tiap part
    parts = []
    for pm in manifest["parts"]:
        p = work / pm["name"]
        if not (p.exists() and p.stat().st_size == pm["size"] and _sha256(p) == pm["sha256"]):
            download_asset(repo, tag, pm["name"], p, token)
            if _sha256(p) != pm["sha256"]:
                raise RuntimeError(f"sha256 mismatch: {pm['name']}")
        parts.append(p)
    print(f"[pull] {len(parts)} part terverifikasi (sha256).")

    # satukan part → tar utuh (bila split)
    tar_path = work / manifest["tar_basename"]
    if manifest.get("split"):
        with open(tar_path, "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    while True:
                        chunk = f.read(8 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
        print(f"[pull] {len(parts)} part disatukan → {tar_path.name}")
    else:
        tar_path = parts[0]

    # extract → data_dir (path di tar relatif: subject/session/frame/file)
    decomp = "zstd -d" if manifest["archive_ext"].endswith("zst") else "gzip -d"
    print(f"[pull] extract → {data_dir} ...")
    r = subprocess.run(["tar", "-C", str(data_dir),
                        "--use-compress-program", decomp, "-xf", str(tar_path)])
    if r.returncode != 0:
        raise RuntimeError(f"extract gagal (rc={r.returncode})")
    have = sum(1 for _ in data_dir.glob("*/*/frame_*/*"))
    print(f"[pull] selesai: {have} file di {data_dir} (manifest {manifest['file_count']}).")
    return manifest


# ───────────────────────── CLI ─────────────────────────

def main():
    ap = argparse.ArgumentParser(description="GitHub Release assets helper (dataset)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = {"--repo": dict(required=True, help="OWNER/REPO"),
              "--tag": dict(required=True, help="tag release, mis. data-v8"),
              "--token": dict(default=os.environ.get("GITHUB_TOKEN"), help="PAT (default env GITHUB_TOKEN)")}

    up = sub.add_parser("upload", help="unggah file sbg aset release")
    up.add_argument("--files", nargs="+", required=True)
    pull = sub.add_parser("pull", help="unduh+extract dataset dari release")
    pull.add_argument("--data_dir", required=True)
    dele = sub.add_parser("delete", help="hapus release + tag (versi lama)")
    for p in (up, pull, dele):
        for flag, kw in common.items():
            p.add_argument(flag, **kw)

    args = ap.parse_args()
    if not args.token:
        sys.exit("Token kosong — set --token atau env GITHUB_TOKEN.")
    if args.cmd == "upload":
        upload_assets(args.repo, args.tag, args.files, args.token)
    elif args.cmd == "pull":
        pull_dataset(args.repo, args.tag, args.data_dir, args.token)
    elif args.cmd == "delete":
        delete_release(args.repo, args.tag, args.token)


if __name__ == "__main__":
    main()
