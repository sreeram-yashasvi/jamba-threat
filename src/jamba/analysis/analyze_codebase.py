import os
import json
from typing import Dict, List
from code_complexity import analyze_file_complexity

def analyze_directory(directory: str, file_pattern: str = "*.py") -> Dict[str, Dict[str, Dict[str, float]]]:
    """Analyze complexity metrics for all Python files in a directory."""
    results = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)
            
            print(f"Analyzing {relative_path}...")
            metrics = analyze_file_complexity(file_path)
            results[relative_path] = metrics
    
    return results

def generate_complexity_report(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """Generate a formatted report of complexity metrics."""
    report = []
    report.append("# Code Complexity Analysis Report\n")
    
    # Calculate average metrics
    total_cyclomatic = 0
    total_files = 0
    total_volume = 0
    total_effort = 0
    max_cyclomatic = 0
    max_cyclomatic_file = ""
    
    for file_path, metrics in results.items():
        if "error" in metrics:
            continue
            
        total_files += 1
        cyclomatic = metrics["cyclomatic"]["complexity"]
        total_cyclomatic += cyclomatic
        
        if "halstead" in metrics:
            total_volume += metrics["halstead"]["volume"]
            total_effort += metrics["halstead"]["effort"]
        
        if cyclomatic > max_cyclomatic:
            max_cyclomatic = cyclomatic
            max_cyclomatic_file = file_path
    
    # Summary section
    report.append("## Summary\n")
    if total_files > 0:
        avg_cyclomatic = total_cyclomatic / total_files
        avg_volume = total_volume / total_files
        avg_effort = total_effort / total_files
        
        report.append(f"- Total files analyzed: {total_files}")
        report.append(f"- Average cyclomatic complexity: {avg_cyclomatic:.2f}")
        report.append(f"- Average Halstead volume: {avg_volume:.2f}")
        report.append(f"- Average Halstead effort: {avg_effort:.2f}")
        report.append(f"- Highest cyclomatic complexity: {max_cyclomatic} (in {max_cyclomatic_file})\n")
    
    # Detailed metrics section
    report.append("## Detailed Metrics\n")
    for file_path, metrics in sorted(results.items()):
        report.append(f"### {file_path}\n")
        
        if "error" in metrics:
            report.append(f"Error: {metrics['error']}\n")
            continue
        
        # Cyclomatic complexity
        cyclomatic = metrics["cyclomatic"]["complexity"]
        report.append(f"#### Cyclomatic Complexity: {cyclomatic}")
        report.append("Interpretation:")
        if cyclomatic <= 5:
            report.append("- Low risk - Simple code")
        elif cyclomatic <= 10:
            report.append("- Moderate risk - Moderately complex")
        else:
            report.append("- High risk - Complex code, consider refactoring")
        report.append("")
        
        # Halstead metrics
        if "halstead" in metrics:
            h = metrics["halstead"]
            report.append("#### Halstead Metrics")
            report.append(f"- Program vocabulary: {h['vocabulary']:.0f}")
            report.append(f"- Program length: {h['length']:.0f}")
            report.append(f"- Volume: {h['volume']:.2f}")
            report.append(f"- Difficulty: {h['difficulty']:.2f}")
            report.append(f"- Effort: {h['effort']:.2f}")
            report.append(f"- Estimated time to program: {h['time']/3600:.2f} hours")
            report.append(f"- Estimated number of bugs: {h['bugs']:.2f}\n")
    
    return "\n".join(report)

def main():
    """Main function to run the analysis."""
    # Get the src directory path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(os.path.dirname(current_dir))
    
    print(f"Analyzing code complexity in {src_dir}...")
    results = analyze_directory(src_dir)
    
    # Generate and save the report
    report = generate_complexity_report(results)
    report_path = os.path.join(src_dir, "complexity_report.md")
    
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"\nAnalysis complete! Report saved to {report_path}")
    
    # Save raw metrics as JSON for potential further analysis
    metrics_path = os.path.join(src_dir, "complexity_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Raw metrics saved to {metrics_path}")

if __name__ == "__main__":
    main() 