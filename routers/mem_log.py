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


def log_memory(label):
    if resource is None:
        print(f"[memory] {label}: (resource module unavailable on Windows/non-POSIX)")
        return
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[memory] {label}: {peak_mb:.1f} MB peak RSS so far")
