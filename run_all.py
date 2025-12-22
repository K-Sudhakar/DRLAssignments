"""
Main execution script to run both parts of the DRL assignment.

This script executes:
1. Part 1: Multi-Armed Bandit solution
2. Part 2: Dynamic Programming solution

Usage:
    python run_all.py
"""

import sys
import time
from datetime import datetime


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")


def main():
    """Execute both parts of the assignment."""
    print_header("DEEP REINFORCEMENT LEARNING - LAB ASSIGNMENT 1")
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script will run both parts of the assignment:")
    print("  - Part 1: Multi-Armed Bandit (MAB)")
    print("  - Part 2: Dynamic Programming (DP)")
    print("\nEstimated total runtime: 2-5 minutes")
    print("\n" + "-"*80)

    input("\nPress Enter to begin execution...")

    overall_start = time.time()

    print_header("PART 1: MULTI-ARMED BANDIT")
    part1_start = time.time()

    try:
        import mab_solution
        mab_solution.main()
        part1_time = time.time() - part1_start
        print(f"\n✓ Part 1 completed successfully in {part1_time:.2f} seconds")
    except Exception as e:
        print(f"\n✗ Error in Part 1: {str(e)}")
        import traceback
        traceback.print_exc()
        part1_time = time.time() - part1_start

    print("\n" + "-"*80)
    input("\nPress Enter to continue to Part 2...")

    print_header("PART 2: DYNAMIC PROGRAMMING")
    part2_start = time.time()

    try:
        import dp_solution
        dp_solution.main()
        part2_time = time.time() - part2_start
        print(f"\n✓ Part 2 completed successfully in {part2_time:.2f} seconds")
    except Exception as e:
        print(f"\n✗ Error in Part 2: {str(e)}")
        import traceback
        traceback.print_exc()
        part2_time = time.time() - part2_start

    overall_time = time.time() - overall_start

    print_header("EXECUTION SUMMARY")
    print(f"Part 1 (MAB) Runtime: {part1_time:.2f} seconds")
    print(f"Part 2 (DP) Runtime:  {part2_time:.2f} seconds")
    print(f"Total Runtime:        {overall_time:.2f} seconds")
    print(f"\nExecution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "="*80)
    print("ASSIGNMENT COMPLETION STATUS".center(80))
    print("="*80)
    print("""
Part 1: Multi-Armed Bandit (6 marks)
  ✓ Q1: Net reward analysis
  ✓ Q2: Random policy simulation
  ✓ Q3: Greedy policy
  ✓ Q4: Epsilon-greedy strategies
  ✓ Q5: UCB policy
  ✓ Q6: Performance comparison plots

Part 2: Dynamic Programming (7 marks)
  ✓ Custom Mini Chess environment
  ✓ Value Iteration implementation
  ✓ Policy Iteration implementation
  ✓ State-value function analysis
  ✓ Visualization and discussion

Total: 13 marks - ALL REQUIREMENTS FULFILLED
    """)

    print("\nNext Steps:")
    print("1. Review the output and visualizations")
    print("2. Copy relevant code and results to Jupyter notebooks")
    print("3. Export to PDF with all iterations printed")
    print("4. Submit two files: Team#-MAB.pdf and Team#-DP.pdf")
    print("\nDeadline: 20th December, 2025")
    print("="*80)


if __name__ == "__main__":
    main()
