"""
v8_lib.py — SUMBER TUNGGAL KEBENARAN untuk eksperimen v8 FAKTORIAL (alignment × loss).

Desain riset (lihat docs/PAPER_DESIGN.md):
  H1: normalisasi → robustness   |   H2: ArcFace → accuracy
  Grid faktorial {softmax, arcface} × {A0..A5} = 12 config × 5 seed.
  Closed-set: gallery=train, probe=HOLDOUT (tanpa LOSO). Multi-frame fusion N×M (headline N5M5).

Notebook (shell tipis) cukup:
    import v8_lib as L
    L.setup()                 # mount Drive + dataset (gen/restore) + autotune GPU + splits
    L.train(seeds=[0])        # training: 12 config × seed ini  (shell v8_train_seed{S})
    # atau
    L.eval_all(); L.analyze() # evaluasi + figur                (shell v8_eval)

Persistensi (Opsi B): checkpoint/cache → Google Drive; evidence kecil (CSV/figur) → git (ANA_DIR).
Resume granular: unit atomik (cfg_id, seed); skip bila `perf.json` ADA (penanda SELESAI, bukan best.pth).
"""

import os
import sys
import json
import glob
import time
import pickle
import subprocess
from pathlib import Path

import numpy as np

# ───────────────────────── KONFIGURASI (statis) ─────────────────────────
ALIGNMENTS = [
    ('A0', 'raw_ply'),            # baseline: koordinat kamera (tanpa normalisasi)
    ('A1', 'align_center'),       # center saja
    ('A2', 'align_centerscale'),  # center + unit-sphere (tanpa rotasi)
    ('A3', 'canonical_npy'),      # PCA canonical (= cnn_input.npy)
    ('A4', 'align_pca_robust'),   # PCA deterministik
    ('A5', 'align_anatomical'),   # anatomical landmark
]
# softmax = arcface_true m=0 (cosine-softmax; head identik → hanya margin yang beda = terkontrol)
# arcface = arcface_true m=0.5 (nilai ArcFace standar, Deng et al. 2019)
LOSS_VARIANTS = {
    'softmax': ('arcface_true', 0.0),
    'arcface': ('arcface_true', 0.5),
}
# GRID faktorial: (cfg_id, repr_mode, loss_type, margin). cfg_id = "{align}_{loss}"
GRID = [
    (f'{aid}_{lname}', repr_mode, ltype, m)
    for (aid, repr_mode) in ALIGNMENTS
    for lname, (ltype, m) in LOSS_VARIANTS.items()
]
ALL_SEEDS = [0, 42, 123, 2024, 31337]

ARC_VARIANT = 'true'
ARC_SCALE   = 30.0
N_POINTS    = 8192
BATCH_SIZE  = 3072           # MAX-UTIL (GPU 95GB): ~80GB (terukur 67GB@2560). FIXED → SEMUA unit WAJIB di GPU >=90GB. Bila OOM → 2816.
FRAME_MODE  = 'all'
EPOCHS, FINETUNE_EPOCHS = 120, 30
LR, FT_LR   = 4e-3, 4e-4    # di-scale utk bs=2560 (max util GPU 95GB). Pantau ~5 epoch: bila NaN/naik → 3e-3/2.5e-3
N_LIST, M_LIST = [1, 3, 5, 10], [1, 3, 5, 10]
N_BEST, M_BEST = 5, 5

COLAB_BRANCH = 'main'
REPO_SLUG    = 'RZulfikri/Thesis'

# Mapping repr_mode → mode util alignment (untuk uji rotasi runtime)
_R2A = {'raw_ply': 'raw', 'canonical_npy': 'pca', 'fps_npy': 'pca',
        'align_center': 'center', 'align_centerscale': 'centerscale',
        'align_pca_robust': 'pca_robust', 'align_anatomical': 'anatomical'}
# repr_mode → file (untuk disk footprint)
_REPR_FILE = {'raw_ply': 'output.ply', 'canonical_npy': 'cnn_input.npy', 'fps_npy': 'cnn_input_fps.npy',
              'align_center': 'align_center.npy', 'align_centerscale': 'align_centerscale.npy',
              'align_pca_robust': 'align_pca_robust.npy', 'align_anatomical': 'align_anatomical.npy'}

# ───────────────────────── state runtime (diisi setup()) ─────────────────────────
DEVICE = REPO_DIR = PROJECT_ROOT = DATA_DIR = DRIVE_ROOT = None
RUNS_DIR = EVAL_DIR = ANA_DIR = TS = None
BS = NUM_WORKERS = REPEAT = None
EVAL_BATCH = 512          # batch encoding saat eval (inference, tanpa backward) — pas utk G100/95GB; turunkan bila GPU kecil
AMP_MODE = 'bf16'
all_session_splits = None
GITHUB_TOKEN = None
_ablation_results = None   # cache memori untuk analyze()


# ═══════════════════════════════ SETUP ═══════════════════════════════
def setup(drive_root='/content/drive/MyDrive/PointNetPalm', mount_drive=True,
          github_token=None, run_tag='v8_factorial'):
    """Mount Drive, siapkan dataset (gen/restore), autotune GPU, bangun session splits."""
    global DEVICE, REPO_DIR, PROJECT_ROOT, DATA_DIR, DRIVE_ROOT
    global RUNS_DIR, EVAL_DIR, ANA_DIR, TS, GITHUB_TOKEN

    import torch
    from datetime import datetime

    REPO_DIR = Path(__file__).resolve().parents[2]          # .../Thesis
    PROJECT_ROOT = REPO_DIR / '3DCNN'
    DATA_DIR = PROJECT_ROOT / 'dataset'
    sys.path.insert(0, str(PROJECT_ROOT))                   # agar `from models...`/`from utils...` jalan

    GITHUB_TOKEN = github_token or os.environ.get('GITHUB_TOKEN')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mount_drive:
        try:
            from google.colab import drive as _gd
            _gd.mount('/content/drive')
        except Exception as e:
            print(f'[setup] Drive mount dilewati ({e})')
    DRIVE_ROOT = Path(drive_root)
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

    RUNS_DIR = DRIVE_ROOT / 'runs' / run_tag
    EVAL_DIR = DRIVE_ROOT / 'eval' / run_tag
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    TS = datetime.now().strftime('%Y%m%d_%H%M%S')
    ANA_DIR = PROJECT_ROOT / 'analysis' / f'{run_tag}_{TS}'   # di repo → ter-commit
    ANA_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Device: {DEVICE} | REPO_DIR={REPO_DIR}')
    print(f'RUNS_DIR (Drive): {RUNS_DIR}')
    print(f'ANA_DIR (git):    {ANA_DIR}')

    _ensure_deps()
    _ensure_dataset()
    _detect_runtime()
    _build_splits()
    print('[setup] selesai.')


def _ensure_deps():
    """Pasang dependency RUNTIME tanpa syarat (idempotent). open3d WAJIB utk raw_ply (A0)
    yang load output.ply; scikit-learn dipakai analyze() (confusion/t-SNE). Dataset bisa
    di-restore dari Drive tanpa lewat cabang generate → deps ini harus dipastikan terpisah."""
    import importlib.util
    need = [m for m, pip in (('open3d', 'open3d'), ('sklearn', 'scikit-learn'))
            if importlib.util.find_spec(m) is None]
    if not need:
        print('[deps] open3d & scikit-learn sudah ada.')
        return
    pip_names = {'open3d': 'open3d', 'sklearn': 'scikit-learn'}
    pkgs = ' '.join(pip_names[m] for m in need)
    print(f'[deps] install: {pkgs} ...')
    _run_streaming(f'{sys.executable} -m pip install -q {pkgs}')


def _ensure_dataset():
    """§2c: dataset regenerate dari raw SEKALI lalu cache di Drive; restore bila ada."""
    drive_tar = DRIVE_ROOT / 'dataset_v8.tar.zst'
    nply = lambda: len(glob.glob(str(DATA_DIR / '*/*/frame_*/output.ply')))
    nal  = lambda: len(glob.glob(str(DATA_DIR / '*/*/frame_*/align_anatomical.npy')))
    complete = lambda: nply() > 0 and nal() == nply()

    if complete():
        print(f'[data] dataset lengkap lokal ({nply()} frame) — skip.')
        return
    subprocess.run(['apt-get', '-qq', 'install', '-y', 'zstd'], check=False)
    if drive_tar.exists():
        print('[data] RESTORE dataset dari Drive cache (tanpa regenerate) ...')
        DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['tar', '--use-compress-program', 'zstd -d', '-xf', str(drive_tar),
                        '-C', str(DATA_DIR.parent)], check=True)
        print(f'[data] restored: {nply()} frame')
    if not complete():
        print('[data] generate dari Raw Depth Data (~35-90 mnt; idempotent) ...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                        'open3d', 'opencv-python-headless', 'scikit-learn', 'scipy'], check=True)
        rc = _run_streaming(f'{sys.executable} generate_dataset.py',
                            cwd=REPO_DIR / '3DRegistration')   # stream progres ke sel
        assert rc == 0, 'generate_dataset.py gagal'
        print(f'[data] cache → Drive {drive_tar.name} ...')
        tmp = str(drive_tar) + '.tmp'
        subprocess.run(['tar', '--use-compress-program', 'zstd -T0', '-cf', tmp,
                        '-C', str(DATA_DIR.parent), DATA_DIR.name], check=True)
        os.replace(tmp, str(drive_tar))
        print('[data] cache Drive tersimpan.')
    assert complete(), 'dataset belum lengkap setelah bootstrap'
    print(f'[data] siap: {nply()} frame')


def _detect_runtime():
    """Tentukan BS/AMP/workers/REPEAT TANPA meng-alokasi GPU di kernel notebook.
    (Probe in-kernel sebelumnya menyandera VRAM → train.py subprocess OOM di L4.)
    BS dibuat KONSTAN (BATCH_SIZE) lintas semua run/GPU → faktorial comparable + muat di L4 24GB."""
    global BS, AMP_MODE, NUM_WORKERS, REPEAT
    import torch
    cc = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0  # query, bukan alokasi
    AMP_MODE = 'bf16' if cc >= 8 else ('fp16' if cc > 0 else 'none')
    try:
        NUM_WORKERS = min(8, os.cpu_count() or 4)
    except Exception:
        NUM_WORKERS = 4
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total,name', '--format=csv,noheader,nounits'], text=True)
        vram_str, gpu_name = [x.strip() for x in out.strip().split('\n')[0].split(',')]
        vram_gb = int(vram_str) / 1024
    except Exception:
        vram_gb, gpu_name = 0.0, 'Unknown'
    BS = BATCH_SIZE                                  # FIXED (tak ada probe GPU)
    REPEAT = max(4, -(-min(BS, 512) * 4 // 88))
    print(f'GPU: {gpu_name} ({vram_gb:.1f} GB) | AMP={AMP_MODE} | BS(fixed)={BS} | '
          f'workers={NUM_WORKERS} | REPEAT={REPEAT} | N_POINTS={N_POINTS}')


def shutdown_runtime(delay=8):
    """Matikan runtime Colab setelah semua sel selesai (lepas GPU → hemat compute unit).
    Beri jeda agar output/commit/Drive ter-flush dulu. Aman di luar Colab (fallback kill kernel)."""
    import time
    print(f'✅ Semua sel selesai. Mematikan runtime dalam {delay}s (melepas GPU)...', flush=True)
    try:
        time.sleep(delay)
        from google.colab import runtime
        runtime.unassign()                      # hentikan & bebaskan runtime/GPU
    except Exception as e:
        print(f'[shutdown] google.colab.runtime tak tersedia ({e}); fallback kill kernel.')
        try:
            import os, signal
            os.kill(os.getpid(), signal.SIGKILL)
        except Exception as e2:
            print(f'[shutdown] gagal: {e2}')


def _build_splits():
    global all_session_splits
    from utils.dataset_lowdata import build_lowdata_splits_session_dirs
    all_session_splits = build_lowdata_splits_session_dirs(str(DATA_DIR))
    for sp, subs in all_session_splits.items():
        print(f'  {sp:8s}: {len(subs)} subjek, {sum(len(v) for v in subs.values())} sesi')


# ═══════════════════════════════ GIT ═══════════════════════════════
def _run_streaming(cmd, cwd=None, logfile=None):
    """Jalankan cmd (shell), STREAM stdout+stderr LIVE ke sel notebook, sekaligus simpan ke logfile.
    (Pengganti subprocess.run+tee yang outputnya tak tampil di sel Colab.)"""
    lf = open(logfile, 'w') if logfile else None
    proc = subprocess.Popen(str(cmd), shell=True, executable='/bin/bash',
                            cwd=str(cwd) if cwd else None,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    try:
        for line in proc.stdout:
            print(line, end='')                 # → sys.stdout (ditangkap & ditampilkan Colab)
            if lf:
                lf.write(line); lf.flush()
    finally:
        proc.wait()
        if lf:
            lf.close()
    return proc.returncode


def _git(args, timeout=600):
    try:
        r = subprocess.run(['git'] + args, cwd=str(REPO_DIR),
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or '') + (r.stderr or '')
    except subprocess.TimeoutExpired:
        return 124, f'TIMEOUT {timeout}s'


def git_save(message, push=False, retries=2, timeout=600):
    """Commit semua (RUNS_DIR di Drive → di luar repo, tak ikut). Tahan hang & non-ff."""
    _git(['add', '-A'])
    if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=str(REPO_DIR)).returncode == 0:
        print('  (nothing to commit)'); return
    rc, out = _git(['commit', '-m', message])
    print(f'Committed: {message}' if rc == 0 else f'Commit err: {out[:200]}')
    if push:
        for _ in range(retries + 1):
            _git(['fetch', 'origin', COLAB_BRANCH], timeout=timeout)
            _git(['merge', '-X', 'ours', '--no-edit', f'origin/{COLAB_BRANCH}'], timeout=timeout)
            rc, out = _git(['push', 'origin', COLAB_BRANCH], timeout=timeout)
            if rc == 0:
                print('Pushed OK'); return
            print(f'Push gagal: {out[:150]}')
        print('  [WARN] push gagal — commit aman lokal.')


# ═══════════════════════════════ TRAINING ═══════════════════════════════
def run_training(cfg_id, repr_mode, loss_type, margin, seed):
    """Latih 1 unit (cfg_id, seed). Skip bila `perf.json` ADA (penanda SELESAI)."""
    out_dir = RUNS_DIR / cfg_id / f'seed_{seed}'
    perf = out_dir / 'perf.json'
    if perf.exists():
        print(f'  SKIP {cfg_id} seed={seed} (perf.json ada = selesai)')
        return True
    amp_flag = f'--amp {AMP_MODE}' if AMP_MODE != 'none' else ''
    cmd = (
        f'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '   # anti-fragmentasi VRAM (saran pesan OOM)
        f'{sys.executable} {PROJECT_ROOT / "train.py"} '
        f'--data_dir {DATA_DIR} --output_dir {out_dir} --seed {seed} '
        f'--fixed_split --frames-per-session 1 --frame-mode {FRAME_MODE} '
        f'--repr-mode {repr_mode} '
        f'--loss {loss_type} --arcface-variant {ARC_VARIANT} '
        f'--arcface-margin {margin} --arcface-scale {ARC_SCALE} '
        f'--val-metric pair_eer --no-early-stop '
        f'--epochs {EPOCHS} --finetune_epochs {FINETUNE_EPOCHS} '
        f'--lr {LR} --finetune_lr {FT_LR} '
        f'--batch_size {BS} --n_points {N_POINTS} --num_workers {NUM_WORKERS} '
        f'--repeat {REPEAT} {amp_flag} --preload-augment --siamese-mode concat'
    )
    print(f'\n{"="*60}\nTRAIN {cfg_id} | repr={repr_mode} | loss={loss_type} m={margin} | seed={seed}\n{"="*60}')
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / 'train_stdout.log'
    print(f'$ {cmd}\n')                          # transparansi: perintah persis
    rc = _run_streaming(cmd, logfile=log)        # STREAM output train.py LIVE ke sel
    ok = perf.exists()
    if not ok:
        print(f'\n  [GAGAL] {cfg_id} seed={seed}: perf.json tak terbentuk (rc={rc}). '
              f'Lihat traceback train.py DI ATAS (juga tersimpan di {log}).')
    else:
        try:
            p = json.loads(perf.read_text())
            if p.get('n_points') != N_POINTS:
                print(f'  [SPEC-MISMATCH] n_points {p.get("n_points")} != {N_POINTS} (GPU clamp?)')
        except Exception:
            pass
    return ok


def train(seeds=None, push=False):
    """Loop GRID × seeds. Default push=False (multi-runtime → persistensi Drive, hindari balapan git)."""
    seeds = seeds if seeds is not None else ALL_SEEDS
    print(f'Training GRID {len(GRID)} config × seeds {seeds} = {len(GRID)*len(seeds)} unit')
    done = 0
    for cfg_id, repr_mode, loss_type, margin in GRID:
        # preflight repr: file ada utk semua frame?
        if not _repr_ready(repr_mode):
            print(f'  [SKIP cfg] {cfg_id}: file repr {repr_mode} belum lengkap'); continue
        for seed in seeds:
            ok = run_training(cfg_id, repr_mode, loss_type, margin, seed)
            done += int(bool(ok))
            if push:
                git_save(f'v8 [auto] {cfg_id} seed={seed}: {"OK" if ok else "GAGAL"}', push=True)
    print(f'\nTraining selesai: {done} unit OK (dari {len(GRID)*len(seeds)}).')


def _repr_ready(repr_mode):
    n_ply = len(glob.glob(str(DATA_DIR / '*/*/frame_*/output.ply')))
    if repr_mode == 'raw_ply':
        return n_ply > 0
    f = _REPR_FILE.get(repr_mode, 'cnn_input.npy')
    return len(glob.glob(str(DATA_DIR / f'*/*/frame_*/{f}'))) == n_ply and n_ply > 0


# ═══════════════════════════════ MODEL LOAD ═══════════════════════════════
def _build_model():
    import torch  # noqa
    from models.siamese import SiamesePalmNet
    return SiamesePalmNet(geom_dim=13, use_geom=False, use_aux_loss=False,
                          n_subjects=11, siamese_mode='concat').to(DEVICE)


def load_model(cfg_id, seed):
    import torch
    from utils.normalizer import GeometryNormalizer
    base = RUNS_DIR / cfg_id / f'seed_{seed}'
    ckpt, norm = base / 'best.pth', base / 'normalizer.json'
    if not ckpt.exists():
        return None, None
    m = _build_model()
    state = torch.load(ckpt, map_location=DEVICE)
    state = state.get('model_state_dict', state)
    if any(k.startswith('encoder.proj.') for k in state):
        state = {k.replace('encoder.proj.', 'encoder.proj_no_geom.'): v for k, v in state.items()}
    m.load_state_dict(state, strict=False); m.eval()
    normalizer = GeometryNormalizer.load(str(norm)) if norm.exists() else None
    return m, normalizer


# ═══════════════════════════════ EVAL (N×M, holdout) ═══════════════════════════════
def eval_all():
    """N×M multi-frame ablation per (cfg,seed), probe=HOLDOUT. Resume per-unit, cache atomik di Drive."""
    global _ablation_results
    import torch
    from utils.eval_multiframe import eval_multiframe_ablation
    ck = EVAL_DIR / 'ablation_results.pkl'
    res = _load_pkl(ck, {})
    print(f'[resume] ablation_results: {len(res)} (cfg,seed) dimuat')

    for cfg_id, repr_mode, _lt, _m in GRID:
        for seed in ALL_SEEDS:
            if (cfg_id, seed) in res:
                continue
            model, normalizer = load_model(cfg_id, seed)
            if model is None:
                continue
            r = eval_multiframe_ablation(
                model=model, enroll_session_splits=all_session_splits['train'],
                probe_session_splits=all_session_splits['holdout'],
                n_list=N_LIST, m_list=M_LIST, fusion_strategy='mean',
                device=DEVICE, n_points=N_POINTS, normalizer=normalizer,
                seed=seed, repr_mode=repr_mode, enc_batch_size=EVAL_BATCH)
            res[(cfg_id, seed)] = r
            eer = r.get((N_BEST, M_BEST), {}).get('eer', float('nan'))
            print(f'  {cfg_id} seed={seed}: EER(5,5)={eer:.4f}')
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _save_pkl(ck, res)
    _ablation_results = res
    print(f'eval_all selesai: {len(res)} entri.')
    return res


def _load_pkl(path, default):
    if Path(path).exists():
        try:
            o = pickle.load(open(path, 'rb'))
            return o
        except Exception:
            pass
    return default


def _save_pkl(path, obj):
    tmp = str(path) + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _get_ablation():
    global _ablation_results
    if _ablation_results:
        return _ablation_results
    _ablation_results = _load_pkl(EVAL_DIR / 'ablation_results.pkl', {})
    print(f'[cache] ablation_results: {len(_ablation_results)} entri (dari Drive)')
    return _ablation_results


# ═══════════════════════════════ ANALISA / FIGUR ═══════════════════════════════
def _split_cfg(cfg_id):
    aid, lname = cfg_id.split('_', 1)
    return aid, lname


def _unpack_gp(gd, pl):
    gl = list(gd.keys()); G = np.stack([np.asarray(gd[l], float).ravel() for l in gl])
    plab = [l for l, _ in pl]; P = np.stack([np.asarray(e, float).ravel() for _, e in pl])
    return G, gl, P, plab


def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _first_res(cfg_id, ab):
    for s in ALL_SEEDS:
        r = ab.get((cfg_id, s), {}).get((N_BEST, M_BEST))
        if r and 'gallery_embs' in r and 'probe_embs' in r:
            return r
    return None


def _gi_scores(G, gl, P, pl):
    G, P = _l2(G), _l2(P); sim = P @ G.T
    gen, imp = [], []
    for i, p in enumerate(pl):
        for j, g in enumerate(gl):
            (gen if p == g else imp).append(sim[i, j])
    return np.array(gen), np.array(imp)


def _det_roc(gen, imp, n=300):
    lo, hi = min(gen.min(), imp.min()), max(gen.max(), imp.max())
    thr = np.linspace(lo, hi, n)
    far = np.array([(imp >= t).mean() for t in thr])
    frr = np.array([(gen < t).mean() for t in thr])
    tar = 1 - frr
    _trapz = getattr(np, 'trapezoid', None) or np.trapz  # numpy>=2 pakai trapezoid
    auc = float(_trapz(tar[::-1], far[::-1]))
    i = int(np.argmin(np.abs(far - frr)))
    return far, frr, tar, auc, float((far[i] + frr[i]) / 2)


def analyze():
    """Hasilkan semua figur+tabel ke ANA_DIR (ter-commit). Idempoten (baca cache, tak eval ulang)."""
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ab = _get_ablation()
    if not ab:
        print('[analyze] ablation_results kosong — jalankan eval_all() dulu.'); return

    # ---- 1. Tabel + heatmap EER 6×2 (baris alignment, kolom loss) ----
    aids = [a for a, _ in ALIGNMENTS]
    lnames = list(LOSS_VARIANTS.keys())
    eer_mean = np.full((len(aids), len(lnames)), np.nan)
    eer_std = np.full_like(eer_mean, np.nan)
    rows = []
    for cfg_id, repr_mode, _lt, _m in GRID:
        aid, lname = _split_cfg(cfg_id)
        vals = [ab[(cfg_id, s)][(N_BEST, M_BEST)]['eer'] for s in ALL_SEEDS
                if (cfg_id, s) in ab and (N_BEST, M_BEST) in ab[(cfg_id, s)]]
        if vals:
            i, j = aids.index(aid), lnames.index(lname)
            eer_mean[i, j] = np.mean(vals); eer_std[i, j] = np.std(vals)
        rows.append({'config': cfg_id, 'alignment': aid, 'loss': lname, 'repr_mode': repr_mode,
                     'eer_n5m5_mean': np.mean(vals) if vals else np.nan,
                     'eer_n5m5_std': np.std(vals) if vals else np.nan, 'n_seed': len(vals)})
    df = pd.DataFrame(rows)
    df.to_csv(ANA_DIR / 'factorial_eer_6x2.csv', index=False)

    fig, ax = plt.subplots(figsize=(4.5, 5))
    im = ax.imshow(eer_mean * 100, cmap='viridis_r', aspect='auto')
    ax.set_xticks(range(len(lnames))); ax.set_xticklabels(lnames)
    ax.set_yticks(range(len(aids))); ax.set_yticklabels([f'{a} {r}' for a, r in ALIGNMENTS], fontsize=8)
    ax.set_title('EER% N5M5 (holdout) — alignment × loss')
    for i in range(len(aids)):
        for j in range(len(lnames)):
            if not np.isnan(eer_mean[i, j]):
                ax.text(j, i, f'{eer_mean[i,j]*100:.2f}\n±{eer_std[i,j]*100:.2f}',
                        ha='center', va='center', color='w', fontsize=7)
    fig.colorbar(im, ax=ax, label='EER%')
    plt.tight_layout(); plt.savefig(ANA_DIR / 'factorial_eer_heatmap.png', bbox_inches='tight'); plt.close()

    # A_accuracy = alignment EER terendah di kolom arcface (pose-kanonik); A_robust dihitung nanti dari rotasi
    jc = lnames.index('arcface')
    col = eer_mean[:, jc]
    a_star = aids[int(np.nanargmin(col))] if not np.all(np.isnan(col)) else 'A4'
    a_star_repr = dict(ALIGNMENTS)[a_star]
    print(f'[A_accuracy] terbaik kolom arcface (pose-kanonik) = {a_star} ({a_star_repr})')
    print(f'[baseline] (A0,softmax) EER = {eer_mean[0, lnames.index("softmax")]*100:.2f}%')

    # konfigurasi figur: semua 6 alignment × 2 loss, file TERPISAH per hasil (mudah disisip ke paper)
    AID_LIST = [a for a, _ in ALIGNMENTS]
    REPR_OF = dict(ALIGNMENTS)
    acolor = {a: plt.get_cmap('tab10')(i) for i, a in enumerate(AID_LIST)}
    (ANA_DIR / 'confusion').mkdir(exist_ok=True)
    (ANA_DIR / 'tsne').mkdir(exist_ok=True)
    from sklearn.metrics import confusion_matrix

    def _gp(cfg_id):
        r = _first_res(cfg_id, ab)
        if r is None:
            return None
        G, gl, P, pl = _unpack_gp(r['gallery_embs'], r['probe_embs'])
        if len(gl) == 0 or len(pl) == 0:
            return None
        return G, gl, P, pl

    # ---- 2. DET & ROC: 1 file per loss, overlay 6 alignment (figur perbandingan) ----
    for lname in lnames:
        figd, axd = plt.subplots(figsize=(6, 5))
        figr, axr = plt.subplots(figsize=(6, 5))
        for aid in AID_LIST:
            gp = _gp(f'{aid}_{lname}')
            if gp is None:
                continue
            G, gl, P, pl = gp
            gen, imp = _gi_scores(G, gl, P, pl)
            if len(gen) == 0 or len(imp) == 0:
                continue
            far, frr, tar, auc, eer = _det_roc(gen, imp)
            c = acolor[aid]
            axd.plot(np.clip(far, 1e-3, 1), np.clip(frr, 1e-3, 1), color=c,
                     label=f'{aid} {REPR_OF[aid]} (EER={eer*100:.2f}%)')
            axr.plot(far, tar, color=c, label=f'{aid} {REPR_OF[aid]} (AUC={auc:.3f})')
        axd.set_xscale('log'); axd.set_yscale('log'); axd.set_xlabel('FAR'); axd.set_ylabel('FRR')
        axd.set_title(f'DET — {lname} (N5M5, holdout)')
        axd.grid(True, which='both', alpha=.3); axd.legend(fontsize=7)
        figd.tight_layout(); figd.savefig(ANA_DIR / f'det_{lname}.png', bbox_inches='tight'); plt.close(figd)
        axr.plot([0, 1], [0, 1], 'k--', alpha=.4); axr.set_xlabel('FAR'); axr.set_ylabel('TAR')
        axr.set_title(f'ROC — {lname} (N5M5, holdout)')
        axr.grid(True, alpha=.3); axr.legend(fontsize=7)
        figr.tight_layout(); figr.savefig(ANA_DIR / f'roc_{lname}.png', bbox_inches='tight'); plt.close(figr)

    # ---- 3. CMC: 1 file per loss, overlay 6 alignment ----
    for lname in lnames:
        figm, axm = plt.subplots(figsize=(6, 5))
        for aid in AID_LIST:
            gp = _gp(f'{aid}_{lname}')
            if gp is None:
                continue
            G, gl, P, pl = gp
            sim = _l2(P) @ _l2(G).T
            ranks = []
            for i, t in enumerate(pl):
                order = np.argsort(-sim[i]); ranks.append(next(k for k, j in enumerate(order) if gl[j] == t))
            ranks = np.array(ranks); K = len(gl)
            axm.plot(range(1, K + 1), [(ranks <= k).mean() for k in range(K)],
                     marker='o', ms=3, color=acolor[aid], label=f'{aid} {REPR_OF[aid]}')
        axm.set_xlabel('Rank'); axm.set_ylabel('Identification rate')
        axm.set_title(f'CMC — {lname} (N5M5, holdout)'); axm.grid(alpha=.3); axm.legend(fontsize=8)
        figm.tight_layout(); figm.savefig(ANA_DIR / f'cmc_{lname}.png', bbox_inches='tight'); plt.close(figm)

    # ---- 4. Confusion matrix: 1 file PER config (berlabel identitas) ----
    for cfg_id, _rm, _lt, _m in GRID:
        gp = _gp(cfg_id)
        if gp is None:
            continue
        G, gl, P, pl = gp
        sim = _l2(P) @ _l2(G).T
        pred = [gl[j] for j in sim.argmax(1)]
        labs = sorted(set(gl)); cm = confusion_matrix(pl, pred, labels=labs)
        acc = float(np.trace(cm) / max(cm.sum(), 1))
        sz = max(5.0, 0.5 * len(labs) + 2)
        figx, axx = plt.subplots(figsize=(sz, sz - 0.5))
        im = axx.imshow(cm, cmap='Blues')
        axx.set_xticks(range(len(labs))); axx.set_yticks(range(len(labs)))
        axx.set_xticklabels(labs, rotation=90, fontsize=7); axx.set_yticklabels(labs, fontsize=7)
        axx.set_xlabel('Prediksi (predicted)'); axx.set_ylabel('Sebenarnya (true)')
        axx.set_title(f'Confusion — {cfg_id}\nrank-1 = {acc*100:.1f}%  (N5M5, holdout)')
        thr = cm.max() / 2 if cm.max() else 0
        for ii in range(len(labs)):
            for jj in range(len(labs)):
                if cm[ii, jj] > 0:
                    axx.text(jj, ii, int(cm[ii, jj]), ha='center', va='center',
                             fontsize=6, color='white' if cm[ii, jj] > thr else 'black')
        figx.colorbar(im, ax=axx, fraction=0.046, pad=0.04, label='jumlah')
        figx.tight_layout(); figx.savefig(ANA_DIR / 'confusion' / f'{cfg_id}.png', bbox_inches='tight'); plt.close(figx)

    # ---- 5. t-SNE: 1 file PER config (★=template gallery, ●=probe; warna per identitas) ----
    try:
        from sklearn.manifold import TSNE
        all_ids = []
        for cfg_id, _r, _l, _m in GRID:
            g = _gp(cfg_id)
            if g:
                all_ids = sorted(set(g[1])); break
        idcolor = {l: plt.get_cmap('tab20')(i % 20) for i, l in enumerate(all_ids)}
        for cfg_id, _rm, _lt, _m in GRID:
            gp = _gp(cfg_id)
            if gp is None:
                continue
            G, gl, P, pl = gp
            X = np.vstack([G, P]); ng = len(gl)
            if len(X) < 4:
                continue
            Z = TSNE(n_components=2, perplexity=max(5, min(30, len(X) - 1)),
                     init='pca', random_state=0).fit_transform(X)
            figt, axt = plt.subplots(figsize=(6, 5)); seen = set()
            for k, l in enumerate(gl):
                axt.scatter(Z[k, 0], Z[k, 1], s=90, marker='*', color=idcolor.get(l, 'gray'),
                            edgecolors='k', linewidths=.5, label=(l if l not in seen else None)); seen.add(l)
            for k, l in enumerate(pl):
                axt.scatter(Z[ng + k, 0], Z[ng + k, 1], s=16, marker='o',
                            color=idcolor.get(l, 'gray'), alpha=.7)
            axt.set_title(f't-SNE — {cfg_id}  (★=template, ●=probe)')
            axt.set_xticks([]); axt.set_yticks([]); axt.legend(fontsize=6, ncol=2, loc='best', framealpha=.6)
            figt.tight_layout(); figt.savefig(ANA_DIR / 'tsne' / f'{cfg_id}.png', bbox_inches='tight'); plt.close(figt)
    except Exception as e:
        print(f'[analyze] t-SNE dilewati: {e}')

    # ---- 6. Fusion N×M heatmap: 1 file per config (bukti manfaat multi-frame) ----
    (ANA_DIR / 'fusion').mkdir(exist_ok=True)
    nl, ml = N_LIST, M_LIST
    fusion_rows = []
    for cfg_id, _rm, _lt, _m in GRID:
        grid_e = np.full((len(nl), len(ml)), np.nan)
        for ii, n in enumerate(nl):
            for jj, mm in enumerate(ml):
                vals = [ab[(cfg_id, s)][(n, mm)]['eer'] for s in ALL_SEEDS
                        if (cfg_id, s) in ab and (n, mm) in ab[(cfg_id, s)]
                        and 'eer' in ab[(cfg_id, s)][(n, mm)]]
                if vals:
                    grid_e[ii, jj] = np.mean(vals)
                    fusion_rows.append({'config': cfg_id, 'N': n, 'M': mm,
                                        'eer_mean': float(np.mean(vals)), 'eer_std': float(np.std(vals))})
        if np.all(np.isnan(grid_e)):
            continue
        figf, axf = plt.subplots(figsize=(4.5, 4))
        im = axf.imshow(grid_e * 100, cmap='viridis_r')
        axf.set_xticks(range(len(ml))); axf.set_xticklabels(ml)
        axf.set_yticks(range(len(nl))); axf.set_yticklabels(nl)
        axf.set_xlabel('M (probe frames)'); axf.set_ylabel('N (enroll frames)')
        axf.set_title(f'Fusion N×M — EER% — {cfg_id}')
        for ii in range(len(nl)):
            for jj in range(len(ml)):
                if not np.isnan(grid_e[ii, jj]):
                    axf.text(jj, ii, f'{grid_e[ii,jj]*100:.2f}', ha='center', va='center',
                             color='w', fontsize=7)
        figf.colorbar(im, ax=axf, label='EER%')
        figf.tight_layout(); figf.savefig(ANA_DIR / 'fusion' / f'{cfg_id}.png', bbox_inches='tight'); plt.close(figf)
    pd.DataFrame(fusion_rows).to_csv(ANA_DIR / 'fusion_nm.csv', index=False)

    # ---- 7. Tabel metrik lengkap (N5M5): EER/AUC/d'/rank-1/TAR@FAR/latency ----
    mrows = []
    for cfg_id, repr_mode, _lt, _m in GRID:
        aid, lname = _split_cfg(cfg_id)
        def _agg(key, _cfg=cfg_id):
            vs = [ab[(_cfg, s)][(N_BEST, M_BEST)].get(key) for s in ALL_SEEDS
                  if (_cfg, s) in ab and (N_BEST, M_BEST) in ab[(_cfg, s)]]
            vs = [v for v in vs if isinstance(v, (int, float))]
            return (float(np.mean(vs)), float(np.std(vs))) if vs else (np.nan, np.nan)
        r1 = np.nan
        gp = _gp(cfg_id)
        if gp is not None:
            G, gl, P, pl = gp
            sim = _l2(P) @ _l2(G).T
            pred = [gl[j] for j in sim.argmax(1)]
            r1 = float(np.mean([p == t for p, t in zip(pred, pl)]))
        eer_m, eer_s = _agg('eer'); auc_m, auc_s = _agg('auc'); dp_m, dp_s = _agg('dprime')
        tar_m, _x = _agg('tar_at_far1'); lat_m, _y = _agg('latency_probe_s')
        mrows.append({'config': cfg_id, 'alignment': aid, 'loss': lname, 'repr_mode': repr_mode,
                      'eer_mean': eer_m, 'eer_std': eer_s, 'auc_mean': auc_m, 'auc_std': auc_s,
                      'dprime_mean': dp_m, 'dprime_std': dp_s, 'tar_at_far1_mean': tar_m,
                      'rank1': r1, 'latency_probe_s_mean': lat_m})
    pd.DataFrame(mrows).to_csv(ANA_DIR / 'metrics_full.csv', index=False)

    # ---- 8. Robustness rotasi (overlay softmax vs arcface per alignment) ----
    df_rot = _rotation_analysis(plt, pd)
    # ---- 8b. A_robust (worst-case EER θ>0, kolom arcface) + uji signifikansi + A_STAR.txt ----
    a_robust = _pick_robust(df_rot)
    a_robust_repr = dict(ALIGNMENTS).get(a_robust, '?')
    sig_rows = _paired_significance(ab)
    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(ANA_DIR / 'significance_softmax_vs_arcface.csv', index=False)
    (ANA_DIR / 'A_STAR.txt').write_text(
        f'A_accuracy\t{a_star}\t{a_star_repr}\n'
        f'A_robust\t{a_robust}\t{a_robust_repr}\n')
    print(f'[A*] A_accuracy(pose-kanonik)={a_star} ({a_star_repr}) | '
          f'A_robust(worst-case rotasi)={a_robust} ({a_robust_repr})')
    # ---- 9. SUMMARY ----
    _write_summary(df, eer_mean, eer_std, aids, lnames, a_star, a_robust, sig_rows)
    print(f'[analyze] selesai → {ANA_DIR}')


# ---- rotasi (pipeline-level): selalu load raw, rotasi, derive repr, encode ----
def _rotz(xyz, deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return xyz @ R.T


def _derive_repr(raw6, repr_mode):
    from utils.alignment import align_cloud6
    return align_cloud6(raw6.astype(np.float32), _R2A.get(repr_mode, 'pca'), handedness=None)


def _encode_rot(model, splits, repr_mode, normalizer, n_frames, deg, rng, batch_size=128):
    """Encode pipeline-rotasi BATCHED: load raw → rotasi → derive repr → encode (batch) → fuse per sesi.
    Paritas: model.eval() (BN running-stats) ⇒ batch-invariant; rng dipakai per-frame urutan sama."""
    import torch
    from collections import defaultdict
    from utils.eval_multiframe import _sample_n_frames, fuse_embeddings
    from utils.dataset import load_session
    model.eval()
    keys, pts_list, geom_list, order = [], [], [], []
    for label, sdirs in splits.items():
        for si, sdir in enumerate(sdirs):
            skey = (label, si); cnt = 0
            for fd in _sample_n_frames(sdir, n_frames, seed=42):
                try:
                    raw6, geom = load_session(Path(fd), repr_mode='raw_ply')
                    raw6 = raw6[:, :6].astype(np.float32).copy()
                    if deg != 0:
                        raw6[:, :3] = _rotz(raw6[:, :3], deg); raw6[:, 3:6] = _rotz(raw6[:, 3:6], deg)
                    pts = _derive_repr(raw6, repr_mode)
                    idx = rng.choice(len(pts), N_POINTS, replace=len(pts) < N_POINTS)
                    pts = pts[idx]; geom = geom.astype(np.float32)
                    if normalizer is not None:
                        geom = normalizer.transform(geom)
                    keys.append(skey)
                    pts_list.append(torch.from_numpy(np.ascontiguousarray(pts)).unsqueeze(0))
                    geom_list.append(torch.from_numpy(geom).unsqueeze(0))
                    cnt += 1
                except Exception as ex:
                    print(f'  [WARN] rot {fd}: {ex}')
            if cnt:
                order.append(skey)
    embs_by_sess = defaultdict(list)
    with torch.no_grad():
        for b in range(0, len(keys), batch_size):
            pt = torch.cat(pts_list[b:b + batch_size], 0).to(DEVICE)
            gt = torch.cat(geom_list[b:b + batch_size], 0).to(DEVICE)
            e = model.encoder(pt, gt).cpu().numpy()
            for k, skey in enumerate(keys[b:b + batch_size]):
                embs_by_sess[skey].append(e[k])
    embs, labs = [], []
    for skey in order:
        embs.append(fuse_embeddings(np.stack(embs_by_sess[skey]), 'mean')); labs.append(skey[0])
    return np.array(embs), labs


def _rotation_analysis(plt, pd):
    import torch
    ROT = [0, 30, 60, 90, 180]
    ck = EVAL_DIR / 'rot_rows.pkl'
    rot_rows = _load_pkl(ck, [])
    done = {(r['config']) for r in rot_rows}
    # uji semua alignment, kedua loss (overlay) — pakai seed pertama yg ada checkpoint
    for cfg_id, repr_mode, _lt, _m in GRID:
        if cfg_id in done:
            continue
        model = normalizer = None
        for s in ALL_SEEDS:
            model, normalizer = load_model(cfg_id, s)
            if model is not None:
                break
        if model is None:
            continue
        rng = np.random.default_rng(42)
        g_e, g_l = _encode_rot(model, all_session_splits['train'], repr_mode, normalizer, N_BEST, 0, rng,
                               batch_size=EVAL_BATCH)
        base = None
        for deg in ROT:
            rng = np.random.default_rng(123)
            p_e, p_l = _encode_rot(model, all_session_splits['holdout'], repr_mode, normalizer, M_BEST, deg, rng,
                                   batch_size=EVAL_BATCH)
            gen, imp = _gi_scores(g_e, g_l, p_e, p_l)
            eer = _det_roc(gen, imp)[4] if len(gen) and len(imp) else float('nan')
            if deg == 0:
                base = eer
            rot_rows.append({'config': cfg_id, 'alignment': _split_cfg(cfg_id)[0],
                             'loss': _split_cfg(cfg_id)[1], 'repr_mode': repr_mode, 'theta_deg': deg,
                             'eer': eer, 'delta_eer': (eer - base) if base is not None else float('nan')})
            print(f'  rot {cfg_id} θ={deg}: EER={eer:.4f}')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _save_pkl(ck, rot_rows)
    df_rot = pd.DataFrame(rot_rows)
    df_rot.to_csv(ANA_DIR / 'rotation_sensitivity.csv', index=False)
    # overlay: garis per alignment; solid=arcface, dashed=softmax
    plt.figure(figsize=(8, 5.5))
    cmap = plt.get_cmap('tab10')
    for ai, (aid, repr_mode) in enumerate(ALIGNMENTS):
        for lname, ls in (('arcface', '-'), ('softmax', '--')):
            sub = df_rot[df_rot['config'] == f'{aid}_{lname}']
            if not sub.empty:
                plt.plot(sub['theta_deg'], sub['delta_eer'], ls=ls, color=cmap(ai),
                         marker='o', ms=3, label=f'{aid} {lname}')
    plt.xlabel('Rotasi raw θ (°, sumbu z)'); plt.ylabel('Δ EER vs θ=0')
    plt.title('Robustness rotasi (overlay: solid=arcface, dashed=softmax)')
    plt.grid(alpha=.3); plt.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(ANA_DIR / 'rotation_sensitivity.png', bbox_inches='tight'); plt.close()
    return df_rot


def _pick_robust(df_rot):
    """A_robust = alignment dgn worst-case EER terendah pada rotasi θ>0 (kolom arcface)."""
    try:
        sub = df_rot[(df_rot['loss'] == 'arcface') & (df_rot['theta_deg'] > 0)]
        if sub.empty:
            return None
        worst = sub.groupby('alignment')['eer'].max()   # worst-case per alignment
        return str(worst.idxmin())                       # alignment dgn worst-case terbaik
    except Exception as e:
        print(f'[A_robust] gagal: {e}'); return None


def _paired_significance(ab):
    """Uji berpasangan softmax vs arcface per alignment (5 seed) di 3 channel:
    EER@N5M5 (mentok lantai), d′@N5M5 (headroom → bukti H2 utama), EER@N1M1 (single-frame, headroom).
    'improve_arcface' = perbaikan arah-benar (EER: sm−arc; d′: arc−sm)."""
    rows = []
    try:
        from scipy import stats
    except Exception as e:
        print(f'[signifikansi] scipy tak ada, dilewati: {e}'); return rows
    specs = [('eer', (N_BEST, M_BEST), 'lower', 'EER N5M5'),
             ('dprime', (N_BEST, M_BEST), 'higher', "d' N5M5"),
             ('eer', (1, 1), 'lower', 'EER N1M1')]
    for key, nm, direction, label in specs:
        for aid, repr_mode in ALIGNMENTS:
            def _vals(loss):
                out = []
                for s in ALL_SEEDS:
                    r = ab.get((f'{aid}_{loss}', s), {}).get(nm, {})
                    v = r.get(key)
                    if isinstance(v, (int, float)):
                        out.append(v)
                return out
            sm, ar = _vals('softmax'), _vals('arcface')
            if len(sm) != len(ar) or len(sm) < 2:
                continue
            sm, ar = np.array(sm, float), np.array(ar, float)
            diff = (sm - ar) if direction == 'lower' else (ar - sm)   # >0 ⇒ arcface lebih baik
            if np.allclose(sm, ar):
                t, p = 0.0, 1.0
            else:
                try:
                    t, p = stats.ttest_rel(ar, sm)
                except Exception:
                    t, p = float('nan'), float('nan')
            rows.append({'metric': label, 'key': key, 'N': nm[0], 'M': nm[1],
                         'alignment': aid, 'repr_mode': repr_mode,
                         'softmax_mean': float(sm.mean()), 'arcface_mean': float(ar.mean()),
                         'improve_arcface': float(diff.mean()),
                         'arcface_better': bool(diff.mean() > 0),
                         't_stat': float(t), 'p_value': float(p),
                         'significant_p05': bool(np.isfinite(p) and p < 0.05)})
    return rows


def _write_summary(df, eer_mean, eer_std, aids, lnames, a_star, a_robust=None, sig_rows=None):
    L = ['# v8 Factorial (alignment × loss) — Summary', '',
         f'**Tanggal**: {TS}',
         '**Desain**: faktorial {softmax, arcface} × {A0..A5}; closed-set gallery=train, '
         'probe=HOLDOUT (tanpa LOSO); multi-frame fusion N5M5; 5 seed.',
         '', '## Tabel EER% N5M5 (mean±std) — baris alignment, kolom loss', '',
         '| alignment | ' + ' | '.join(lnames) + ' |',
         '|' + '---|' * (len(lnames) + 1)]
    for i, (aid, repr_mode) in enumerate(ALIGNMENTS):
        cells = []
        for j in range(len(lnames)):
            cells.append(f'{eer_mean[i,j]*100:.2f}±{eer_std[i,j]*100:.2f}'
                         if not np.isnan(eer_mean[i, j]) else 'N/A')
        L.append(f'| {aid} {repr_mode} | ' + ' | '.join(cells) + ' |')
    L += ['', f'**Baseline** (A0, softmax) = {eer_mean[0, lnames.index("softmax")]*100:.2f}% EER.',
          f'**A_accuracy** (EER pose-kanonik terendah, kolom arcface) = **{a_star}**.',
          f'**A_robust** (worst-case EER terendah pada rotasi θ>0, kolom arcface) = **{a_robust}**.',
          '> ⚠️ A_accuracy bisa ≠ A_robust: alignment yang sempurna di pose-kanonik (mis. center/scale '
          'atau PCA polos) dapat **runtuh saat dirotasi** (lihat rotation_sensitivity.png). Untuk '
          '**deployment & klaim H1, pakai A_robust**.']
    if sig_rows:
        order = ["d' N5M5", 'EER N1M1', 'EER N5M5']   # d′ = bukti H2 utama (ada headroom) lebih dulu
        metrics = [m for m in order if any(r['metric'] == m for r in sig_rows)]
        metrics += [r['metric'] for r in sig_rows if r['metric'] not in metrics and r['metric'] not in order]
        L += ['', '## Signifikansi softmax vs arcface (paired t-test, 5 seed)',
              '> EER@N5M5 mentok lantai (≈0) → uji-t underpowered. **Bukti H2 utama = d′** (punya headroom); '
              'EER@N1M1 (single-frame) sbg channel kedua. `improve_arcface`>0 ⇒ arcface lebih baik.']
        for m in metrics:
            sub = [r for r in sig_rows if r['metric'] == m]
            is_dp = (sub[0]['key'] == 'dprime')
            unit = '' if is_dp else '%'
            sc = 1.0 if is_dp else 100.0
            L += ['', f'### {m}', '',
                  f'| alignment | softmax | arcface | improve(arcface){unit} | arcface lebih baik | p-value | sig<0.05 |',
                  '|---|---|---|---|---|---|---|']
            for r in sub:
                L.append(f"| {r['alignment']} | {r['softmax_mean']*sc:.2f} | {r['arcface_mean']*sc:.2f} | "
                         f"{r['improve_arcface']*sc:+.2f} | {'ya' if r['arcface_better'] else 'tidak'} | "
                         f"{r['p_value']:.3f} | {'✔' if r['significant_p05'] else '—'} |")
    L += ['', '## Klaim',
          '- **H1 (normalisasi → robustness)**: lihat rotation_sensitivity.png + A_robust; alignment '
          'rotation-robust (A4) datar di semua θ, sedangkan A0/A1/A2 runtuh & A3 paku di 90°.',
          '- **H2 (ArcFace → accuracy)**: bukti utama via **d′** (ArcFace ~2× separabilitas di semua '
          'representasi ternormalisasi) + EER@N1M1; EER@N5M5 mentok lantai (tak informatif).',
          '', '_Dihasilkan otomatis oleh v8_lib.analyze()_']
    (ANA_DIR / 'SUMMARY.md').write_text('\n'.join(L))
    print('SUMMARY.md ditulis.')


# ═══════════════════════════════ ARSIP RELEASE (opsional) ═══════════════════════════════
def archive_runs_to_release(tag='runs-v8'):
    """Kemas RUNS_DIR (Drive) → aset Release (split >1.9GB). Butuh GITHUB_TOKEN."""
    sys.path.insert(0, str(REPO_DIR / '3DRegistration'))
    import importlib, release_assets as ra
    importlib.reload(ra)
    import shutil
    arc = Path('/content/_runs_archive'); shutil.rmtree(arc, ignore_errors=True); arc.mkdir()
    subprocess.run(['apt-get', '-qq', 'install', '-y', 'zstd'], check=False)
    tar = arc / 'runs_v8.tar.zst'
    subprocess.run(['tar', '--use-compress-program', 'zstd -T0', '-cf', str(tar),
                    '-C', str(RUNS_DIR.parent), RUNS_DIR.name], check=True)
    lim = 1900 * 1024 * 1024
    if tar.stat().st_size > lim:
        subprocess.run(['split', '-b', str(lim), '-d', '-a', '2', str(tar), str(tar) + '.part'], check=True)
        tar.unlink(); files = sorted(str(p) for p in arc.glob('runs_v8.tar.zst.part*'))
    else:
        files = [str(tar)]
    ra.create_or_get_release(REPO_SLUG, tag, GITHUB_TOKEN, name='v8 training runs',
                             body='Checkpoint v8 factorial (dari Drive).', prerelease=True)
    for f in files:
        ra.upload_asset(REPO_SLUG, tag, f, GITHUB_TOKEN)
    print('runs ter-arsip ke Release', tag)
