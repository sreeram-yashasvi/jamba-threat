#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import logging
import json

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from jamba.analysis.fault_tree import FaultTreeAnalyzer

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path("fault_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize the analyzer
    analyzer = FaultTreeAnalyzer(output_dir=str(output_dir))
    
    # Build and analyze the fault tree
    logger.info("Building fault tree...")
    root = analyzer.build_fault_tree()
    
    logger.info("Analyzing fault tree...")
    analysis = analyzer.analyze_fault_tree()
    
    # Generate and print the report
    logger.info("Generating report...")
    report = analyzer.generate_report()
    print("\nFault Tree Analysis Report:")
    print("=" * 30)
    print(report)
    
    # Example of system monitoring
    logger.info("Monitoring system metrics...")
    
    # Simulate some metrics
    model_metrics = {
        "accuracy": 0.87,  # Below threshold
        "loss": 1.5,      # Above threshold
        "f1_score": 0.85
    }
    
    resource_metrics = {
        "memory_usage": 0.95,  # High usage
        "gpu_memory": 0.88,    # Below threshold
        "cpu_usage": 0.75
    }
    
    warnings = analyzer.monitor_system(model_metrics, resource_metrics)
    
    if warnings:
        print("\nSystem Warnings:")
        print("=" * 30)
        for warning in warnings:
            print(f"⚠️  {warning}")
    
    # Save metrics for reference
    metrics_file = output_dir / "system_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            "model_metrics": model_metrics,
            "resource_metrics": resource_metrics,
            "warnings": warnings
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 