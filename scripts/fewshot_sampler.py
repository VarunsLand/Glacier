"""CLI Tool to test and verify Few-Shot Episodic Sampling on DL4GAM."""
import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import DL4GAMDataset
from src.sampler import FewShotEpisodeSampler


def main():
    parser = argparse.ArgumentParser(description="Test Few-Shot Glacier Episode Sampler")
    parser.add_argument("--k-shot", type=int, default=5, help="Number of support shots (K)")
    parser.add_argument("--q-queries", type=int, default=2, help="Number of query patches (Q)")
    parser.add_argument("--fold", type=str, default="fold_train", choices=["fold_train", "fold_valid", "fold_test"])
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\nInitializing DL4GAMDataset for fold '{args.fold}'...")
    ds = DL4GAMDataset(root=ROOT, fold=args.fold)
    print(f"Loaded {len(ds)} patches across {len(ds.get_all_glacier_ids())} distinct glaciers.")

    sampler = FewShotEpisodeSampler(
        dataset=ds,
        k_shot=args.k_shot,
        q_queries=args.q_queries,
        seed=args.seed,
    )

    print(f"\nSampling {args.episodes} episodes (K={args.k_shot}, Q={args.q_queries})...")
    for i in range(args.episodes):
        ep = sampler.sample_episode(episode_idx=i)
        print(f"\nEpisode {i+1}:")
        print(f"  Support X: {tuple(ep.support_x.shape)} | Support Y: {tuple(ep.support_y.shape)}")
        print(f"  Query X:   {tuple(ep.query_x.shape)} | Query Y:   {tuple(ep.query_y.shape)}")
        print(f"  Support Glaciers ({len(ep.support_glaciers)}): {ep.support_glaciers[:5]}...")
        print(f"  Query Glaciers   ({len(ep.query_glaciers)}): {ep.query_glaciers[:5]}...")
        
        # Verify strict disjointness
        overlap = set(ep.support_glaciers) & set(ep.query_glaciers)
        assert len(overlap) == 0, f"LEAKAGE DETECTED: {overlap}"
        print("  Disjointness check: PASSED (Zero Glacier Overlap)")

    print("\nSampler verification successful!")


if __name__ == "__main__":
    main()
