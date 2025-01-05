# Option Pricing Methods Performance Analyzer

## Description

This project implements and compares different numerical methods for pricing options in multiple dimensions. It includes three main approaches:

- BSDE-DNN (Backward Stochastic Differential Equations with Deep Neural Networks)
- Longstaff-Schwartz Method
- Finite Difference Method

The analyzer provides comprehensive performance metrics including:

- Price accuracy across different dimensions (2D to 100D)
- Computation time analysis
- Greeks calculations
- Numerical stability analysis
- Error analysis with confidence intervals

## Features

- Multi-dimensional option pricing up to 100 dimensions
- Correlated asset path generation
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Visualization of results using matplotlib and seaborn
- Performance benchmarking across methods
- Stability analysis with confidence intervals

## Installation

### Prerequisites

- Python 3.12
- CUDA-capable GPU (optional, for faster DNN training)

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/option-pricing-analyzer.git
cd option-pricing-analyzer
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
\venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

To run the complete analysis:

```bash
python main.py
```

This will generate several plots:

- Price comparison across dimensions
- Computation time analysis
- Greeks visualization
- Stability analysis
- Error analysis with confidence intervals

### Customizing Parameters

You can modify the base parameters in the `PerformanceAnalyzer` class:

- Initial stock price (S0)
- Strike price (K)
- Risk-free rate (r)
- Volatility (sigma)
- Time to maturity (T)
- Number of time steps
- Number of simulation paths

### Example Code

```python
analyzer = PerformanceAnalyzer()

# Run comprehensive analysis
analyzer.comprehensive_analysis()

# Run specific analyses
stability_results = analyzer.analyze_stability(n_trials=10)
precision_results, timing_results = analyzer.test_precision()
```

## Output

The analysis generates several visualizations:

1. Price comparison plot showing option prices across dimensions for each method
2. Computation time plot (log scale) showing scalability
3. Bar plot of Greeks values
4. Stability analysis plot with error bars
5. Relative error analysis plot with confidence intervals

## Notes

- The Finite Difference method is currently implemented only for 2D cases
- BSDE-DNN training parameters can be adjusted in the `_test_bsde_dnn` method
- GPU acceleration is automatically used if available

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
