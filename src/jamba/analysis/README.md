# Jamba Threat Detection - Fault Tree Analysis

This module provides comprehensive fault tree analysis capabilities for the Jamba Threat Detection system. It helps identify, analyze, and monitor potential failure modes in the system.

## Overview

The fault tree analysis module consists of the following components:

1. `FaultEvent` - A dataclass representing a fault event in the system
2. `FaultTreeAnalyzer` - The main class for building and analyzing fault trees
3. Test suite for validating the analysis functionality
4. Example script demonstrating usage

## Features

- Hierarchical fault tree construction
- Risk metrics calculation
- Probability analysis using OR-gate logic
- Detailed report generation (Markdown and JSON formats)
- System monitoring with configurable thresholds
- Comprehensive test coverage

## Usage

### Basic Usage

```python
from jamba.analysis.fault_tree import FaultTreeAnalyzer

# Initialize the analyzer
analyzer = FaultTreeAnalyzer(output_dir="fault_analysis")

# Generate and print the analysis report
report = analyzer.generate_report()
print(report)
```

### System Monitoring

```python
# Monitor system metrics
model_metrics = {
    "accuracy": 0.95,
    "loss": 0.5
}

resource_metrics = {
    "memory_usage": 0.7,
    "gpu_memory": 0.6
}

warnings = analyzer.monitor_system(model_metrics, resource_metrics)
for warning in warnings:
    print(f"Warning: {warning}")
```

## Fault Tree Structure

The fault tree is organized into three main categories:

1. Data-related failures
   - Data corruption
   - Preprocessing failures
   - Data distribution drift

2. Model-related failures
   - Training failures
   - Inference failures
   - Model performance drift

3. Resource-related failures
   - Memory exhaustion
   - GPU failures
   - I/O failures

## Risk Metrics

The analyzer calculates several risk metrics for each fault event:

- Risk Score = Severity × Probability
- Cumulative Probability (for events with children)
- Raw Probability
- Severity (1-5 scale)

## Report Format

The generated reports include:

1. High-risk events (Risk Score > 0.5)
   - Event name and ID
   - Description
   - Risk metrics
   - Mitigation strategies

2. Complete event listing
   - All events sorted by risk score
   - Detailed metrics
   - Parent-child relationships

## Running Tests

To run the test suite:

```bash
python -m unittest src/jamba/tests/test_fault_tree.py
```

## Example Script

An example script is provided in `examples/fault_tree_example.py`. To run it:

```bash
python examples/fault_tree_example.py
```

This will:
1. Build and analyze a fault tree
2. Generate a detailed report
3. Simulate system monitoring
4. Save results to the specified output directory

## Contributing

When adding new fault events or modifying the analysis:

1. Update the `build_fault_tree()` method
2. Add appropriate test cases
3. Update documentation
4. Ensure all tests pass

## Future Improvements

Planned enhancements:

1. Dynamic threshold adjustment based on historical data
2. Integration with monitoring systems
3. Visualization of fault trees
4. Real-time monitoring capabilities
5. Custom fault event definition support 