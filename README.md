# Deep Reinforcement Learning Assignment 1

**BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, PILANI**
**Work Integrated Learning Programmes Division**

This repository contains complete implementations for Lab Assignment 1, covering:
- **Part 1**: Multi-Armed Bandit Algorithms for Product Recommendation (6 marks)
- **Part 2**: Dynamic Programming for Mini Chess Game (7 marks)

## Assignment Overview

### Part 1: Multi-Armed Bandit (MAB)
Implement and compare various MAB algorithms for e-commerce product recommendations:
- Random Policy (baseline)
- Greedy Policy
- Epsilon-Greedy Policy (multiple exploration rates)
- Upper Confidence Bound (UCB) Policy

**Objective**: Maximize long-term net profit from product recommendations.

### Part 2: Dynamic Programming (DP)
Implement Value Iteration and Policy Iteration for a simplified chess endgame:
- Custom Mini Chess Environment (King + Pawn vs King)
- State space enumeration
- Optimal policy computation
- Value function visualization

**Objective**: Learn optimal play in a tractable MDP.

## Project Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── mab_solution.py          # Part 1: Multi-Armed Bandit implementation
├── dp_solution.py           # Part 2: Dynamic Programming implementation
├── run_all.py               # Execute both parts
└── docs/
    ├── MAB_Report.md        # Detailed report for Part 1
    └── DP_Report.md         # Detailed report for Part 2
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Individual Parts

**Part 1 - Multi-Armed Bandit:**
```bash
python mab_solution.py
```

**Part 2 - Dynamic Programming:**
```bash
python dp_solution.py
```

### Running Both Parts
```bash
python run_all.py
```

## Implementation Details

### Part 1: Multi-Armed Bandit

#### Classes Implemented:
1. **MultiArmedBandit**: Environment managing 6 products and 498 user sessions
2. **RandomPolicy**: Baseline random selection
3. **GreedyPolicy**: Exploit best-performing product after initial exploration
4. **EpsilonGreedyPolicy**: Balance exploration/exploitation with epsilon parameter
5. **UCBPolicy**: Intelligent exploration based on uncertainty

#### Key Features:
- Net reward computation (Revenue - Cost)
- Comprehensive performance comparison
- Cumulative profit visualization
- Statistical analysis of product performance

#### Questions Answered:
- **Q1**: Net reward analysis and product identification
- **Q2**: Random policy simulation (300+ rounds)
- **Q3**: Greedy policy with early lock-in analysis
- **Q4**: Epsilon-greedy with 2%, 10%, 25% exploration rates
- **Q5**: UCB policy demonstrating intelligent exploration
- **Q6**: Comprehensive comparison and visualization

### Part 2: Dynamic Programming

#### Classes Implemented:
1. **MiniChessEnv**: Custom chess environment with:
   - Board size: 4x4 or 5x5 (based on student ID)
   - Pieces: White King + Pawn vs Black King
   - Legal move generation
   - Check, checkmate, stalemate detection
   - Pawn promotion handling

2. **ValueIteration**: Computes optimal value function V*(s)
3. **PolicyIteration**: Computes optimal policy π*(s)

#### Key Features:
- State space enumeration using BFS
- Convergence monitoring
- Value function visualization (heatmaps)
- Policy demonstration
- Performance metrics tracking

#### Deliverables:
1. Custom environment with reset(), step(), render()
2. Both DP algorithms with convergence statistics
3. State-value function heatmaps
4. Analysis of curse of dimensionality
5. Discussion on scalability to full chess

## Results Summary

### Part 1: MAB Results

**Product Performance:**
- Best Product: Identified through mean reward analysis
- Worst Product: Identified through negative profit patterns

**Strategy Comparison:**
- Random Policy: Baseline performance
- Greedy: Risk of premature convergence
- Epsilon-Greedy (10%): Best balance for this dataset
- UCB: Intelligent exploration, often optimal

**Recommendation**: Implement UCB or ε-Greedy (10%) for production

### Part 2: DP Results

**Convergence:**
- Value Iteration: ~50-100 iterations
- Policy Iteration: ~5-15 iterations (fewer but more expensive)

**State Space:**
- 4x4 board: ~2000-5000 reachable states
- 5x5 board: ~5000-10000 reachable states

**Key Insights:**
- DP provides exact solutions for small MDPs
- State space grows exponentially with complexity
- Function approximation needed for large problems

## Assignment Requirements Fulfilled

### Part 1 (6 marks):
- ✅ Q1: Net reward computation and product analysis (1 mark)
- ✅ Q2: Random policy simulation (1 mark)
- ✅ Q3: Greedy policy implementation (0.5 mark)
- ✅ Q4: Epsilon-greedy with multiple rates (2.5 marks)
- ✅ Q5: UCB policy analysis (0.5 mark)
- ✅ Q6: Comprehensive comparison plots (0.5 mark)

### Part 2 (7 marks):
- ✅ Custom environment implementation (1 mark)
- ✅ Value & Policy Iteration (2 marks)
- ✅ State-value function visualization (2 marks)
- ✅ Analysis & discussion (2 marks)

## Code Quality

All code includes:
- ✅ Comprehensive docstrings for every function
- ✅ Type hints for parameters and returns
- ✅ Inline comments explaining key operations
- ✅ Proper error handling
- ✅ Clean, modular architecture
- ✅ Following PEP 8 style guidelines

## Visualizations

The implementation generates:
1. **MAB**: Cumulative profit curves for all strategies
2. **MAB**: Net reward distribution histograms
3. **DP**: State-value function heatmaps
4. **DP**: Board state visualizations

## Performance

**Part 1**: Runs in ~5-10 seconds for 500 rounds
**Part 2**: Runs in ~30-120 seconds depending on board size

## Future Enhancements

### Part 1:
- Thompson Sampling implementation
- Contextual bandits with user features
- Non-stationary environment handling

### Part 2:
- Function approximation with neural networks
- Monte Carlo Tree Search
- Integration with actual chess engines

## References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction
2. Lecture materials: CS1-CS5
3. Assignment specifications and webinar demonstrations

## Authors

Deep Reinforcement Learning Assignment Solution
December 2025

## License

This code is submitted as part of academic coursework for BITS Pilani.

## Contact

For questions or clarifications:
- Pooja Harde: pooja.harde@wilp.bits-pilani.ac.in
- Divya K: divyak@wilp.bits-pilani.ac.in
- Dincy R Arikkat: dincyrarikkat@wilp.bits-pilani.ac.in
