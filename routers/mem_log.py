"""
Shared memory-logging helper. Pure stdlib (resource module) — no new
dependency, and it's temporary diagnostic tooling, not a permanent
feature, so it's kept in its own tiny file rather than mixed into
gemini_shared.py or any router.

resource.getrusage(...).ru_maxrss is a HIGH-WATER MARK: it only ever
goes up for the life of the process, never down (even after Python's
own garbage collector frees objects — the OS doesn't necessarily hand
that memory back to the kernel, which is exactly the "one-way ratchet"
behavior C-extension-heavy libraries like numpy/pandas/sklearn/
onnxruntime/matplotlib exhibit in practice). That makes it ideal here:
each printed number tells you the worst this process has used SO FAR,
so watching it after each feature's lazy-loader finishes shows exactly
which one pushes the total closer to Render's 512MB ceiling.

On Linux (Render's containers), ru_maxrss is reported in KB — divided
by 1024 below to print MB instead.
"""

try:
    import resource
except ImportError:
    resource = None


import os

TOTAL_RAM_LIMIT_MB = 512.0
FINBERT_MEMORY_THRESHOLD_MB = 150.0


def log_memory(label):
    if resource is None:
        print(f"[memory] {label}: (resource module unavailable on Windows/non-POSIX)")
        return
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[memory] {label}: {peak_mb:.1f} MB peak RSS so far")


def get_memory_stats(threshold_mb=FINBERT_MEMORY_THRESHOLD_MB):
    limit_mb = TOTAL_RAM_LIMIT_MB
    env_limit = os.environ.get("RENDER_MEMORY_LIMIT_MB") or os.environ.get("TOTAL_RAM_MB")
    if env_limit:
        try:
            limit_mb = float(env_limit)
        except ValueError:
            pass

    peak_rss_mb = 0.0
    if resource is not None:
        # ru_maxrss is in KB on Linux
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    else:
        # On Windows or non-POSIX
        try:
            import psutil
            peak_rss_mb = psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            peak_rss_mb = 0.0

    available_mb = max(0.0, limit_mb - peak_rss_mb) if peak_rss_mb > 0 else limit_mb
    can_load_finbert = available_mb >= threshold_mb

    if can_load_finbert:
        reason = f"Sufficient memory ({available_mb:.1f} MB free, {threshold_mb:.0f} MB required)"
    else:
        reason = (
            f"Low backend memory ({peak_rss_mb:.1f} MB / {limit_mb:.0f} MB used, "
            f"only {available_mb:.1f} MB free; FinBERT requires ~{threshold_mb:.0f} MB)"
        )

    return {
        "peak_rss_mb": round(peak_rss_mb, 1),
        "limit_mb": round(limit_mb, 1),
        "available_mb": round(available_mb, 1),
        "required_mb": threshold_mb,
        "can_load_finbert": can_load_finbert,
        "reason": reason,
    }

