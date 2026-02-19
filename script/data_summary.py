"""
Print per-task data summary: number of classes, total samples, samples per class.
Helps explain why some tasks (e.g. Task 4) may have lower accuracy.
"""
import sys
import os
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import SpectrumDataLoader


def main():
    base = Path(__file__).resolve().parent.parent
    os.chdir(base)

    print("=" * 60)
    print("Dataset summary by task (classes, samples, balance)")
    print("=" * 60)

    for task_id in [1, 2, 3, 4]:
        try:
            loader = SpectrumDataLoader(data_dir="datasets", task_id=task_id)
            X, y, _ = loader.load_all_data()
            unique, counts = zip(*sorted(Counter(y).items()))
            n_classes = len(unique)
            n_samples = len(y)
            min_per_class = min(counts)
            max_per_class = max(counts)
            mean_per_class = n_samples / n_classes if n_classes else 0
            print(f"\nTask {task_id}:")
            print(f"  Classes:     {n_classes}")
            print(f"  Total:       {n_samples} samples")
            print(f"  Per class:   min={min_per_class}, max={max_per_class}, mean={mean_per_class:.1f}")
            if n_classes <= 20:
                print(f"  Class names: {list(unique)}")
            else:
                print(f"  Class names: {list(unique)[:10]}... (+{n_classes - 10} more)")
        except Exception as e:
            print(f"\nTask {task_id}: ERROR - {e}")

    print("\n" + "=" * 60)
    print("Reasons a task may have lower accuracy:")
    print("  - More classes -> harder multi-class classification")
    print("  - Fewer samples per class -> less data to learn")
    print("  - Similar spectra between classes -> harder to discriminate")
    print("  - Class imbalance -> model biased toward majority")
    print("=" * 60)


if __name__ == "__main__":
    main()
