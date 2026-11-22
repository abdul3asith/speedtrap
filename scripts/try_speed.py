from benchmarks.models import get_model
from benchmarks.policy import load_policy
from benchmarks.speed import measure_speed
from benchmarks.techniques import get_technique

bundle = load_policy(get_model("smolvla"), get_technique("native"))

result = measure_speed(bundle, n=10, warmup=2)

print(f"p50: {result['p50_ms']:.0f} ms   p99: {result['p99_ms']:.0f} ms")
print(
    f"mean: {result['mean_ms']:.0f} ms   jitter: {result['jitter_ms']:.0f} ms   hz: {result['hz']:.2f}"
)
print("each run (ms):", [round(t) for t in result["times_ms"]])
