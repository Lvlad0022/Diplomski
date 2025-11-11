import numpy as np
import pytest
from q_logic_memory_classes import SumTree, ExperienceMemory, TDPriorityReplayBuffer  # prilagodi import!
import traceback
# ====================================================
#                SUMTREE TESTOVI
# ====================================================

def test_experience_memory_sampling_distribution():
    np.random.seed(42)
    n_items = 100
    mem = ExperienceMemory(
        capacity=n_items,
        priorities=True,
        alpha_start=1.0, alpha_end=1.0,
        beta_start=0.0, beta_end=0.0,  # IS težine ne utječu
        segment=False
    )

    # ubacujemo uzorke s prioritetima 1..100
    for i in range(1, n_items + 1):
        mem.push((i, i, i, i, False), priority=float(i))

    # teorijske vjerojatnosti (P(i) ∝ priority^α)
    priorities = np.arange(1, n_items + 1, dtype=np.float64)
    probs_expected = priorities / priorities.sum()

    # uzorkuj 100_000 puta
    n_samples = 2_000
    _, idxs, _ = mem.sample(batch_size=1)
    all_samples = []
    for _ in range(n_samples):
        _, idxs, _ = mem.sample(batch_size=512)
        all_samples.extend(idxs)
    all_samples = np.array(all_samples)

    # broj pojavljivanja svakog indeksa
    counts = np.bincount(all_samples, minlength=n_items)
    probs_empirical = counts / counts.sum()

    # usporedi empirijske i teorijske vjerojatnosti
    corr = np.corrcoef(probs_empirical, probs_expected)[0, 1]
    diff = np.abs(probs_empirical - probs_expected).mean()

    print(f"Pearson correlation: {corr:.4f}, mean abs diff: {diff:.4f}")

    assert corr > 0.98, f"Distribucija ne korelira dovoljno s očekivanom (r={corr:.3f})"
    assert diff < 0.005, f"Distribucija se previše razlikuje (MAE={diff:.3f})"



# ---------------------------
# POKRETANJE TESTOVA RUČNO
# ---------------------------

if __name__ == "__main__":
    tests = [
    test_experience_memory_sampling_distribution
]


    print("🔍 Pokrećem testove ručno...\n")
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__} prošao")
        except Exception as e:
            print(f"❌ {test.__name__} pao:")
            traceback.print_exc()
        print("-" * 60)
