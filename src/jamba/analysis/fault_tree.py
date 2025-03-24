#!/usr/bin/env python3
import logging
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import torch
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FaultEvent:
    """Represents a fault event in the fault tree"""
    id: str
    name: str
    description: str
    severity: int  # 1-5, where 5 is most severe
    probability: float  # 0-1
    mitigation: str
    children: List['FaultEvent'] = None
    parent: Optional['FaultEvent'] = None

class FaultTreeAnalyzer:
    """Analyzes potential failure modes in the Jamba threat detection system"""
    
    def __init__(self, output_dir: str = "fault_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fault_tree = None
        
    def build_fault_tree(self) -> FaultEvent:
        """Build the complete fault tree for the system"""
        # Top level system failure
        root = FaultEvent(
            id="SYS_FAIL",
            name="System Failure",
            description="Complete failure of threat detection system",
            severity=5,
            probability=0.01,
            mitigation="Implement comprehensive monitoring and fallback systems"
        )
        
        # Data-related failures
        data_failure = FaultEvent(
            id="DATA_FAIL",
            name="Data Processing Failure",
            description="Failures in data preprocessing and validation",
            severity=4,
            probability=0.05,
            mitigation="Implement robust data validation and error handling"
        )
        
        data_failure.children = [
            FaultEvent(
                id="DATA_CORRUPT",
                name="Data Corruption",
                description="Input data is corrupted or invalid",
                severity=4,
                probability=0.02,
                mitigation="Add data integrity checks and validation"
            ),
            FaultEvent(
                id="PREPROC_FAIL",
                name="Preprocessing Failure",
                description="Failure in data preprocessing pipeline",
                severity=3,
                probability=0.03,
                mitigation="Add error handling in preprocessing steps"
            ),
            FaultEvent(
                id="DATA_DRIFT",
                name="Data Distribution Drift",
                description="Change in data distribution over time",
                severity=3,
                probability=0.1,
                mitigation="Implement drift detection and model retraining"
            )
        ]
        
        # Model-related failures
        model_failure = FaultEvent(
            id="MODEL_FAIL",
            name="Model Failure",
            description="Failures in model training or inference",
            severity=5,
            probability=0.03,
            mitigation="Implement model validation and monitoring"
        )
        
        model_failure.children = [
            FaultEvent(
                id="TRAIN_FAIL",
                name="Training Failure",
                description="Model training process fails",
                severity=4,
                probability=0.05,
                mitigation="Add checkpointing and training validation"
            ),
            FaultEvent(
                id="INFER_FAIL",
                name="Inference Failure",
                description="Model inference produces invalid results",
                severity=5,
                probability=0.02,
                mitigation="Add inference validation and fallback logic"
            ),
            FaultEvent(
                id="MODEL_DRIFT",
                name="Model Performance Drift",
                description="Model performance degrades over time",
                severity=4,
                probability=0.15,
                mitigation="Implement performance monitoring and retraining triggers"
            )
        ]
        
        # Resource-related failures
        resource_failure = FaultEvent(
            id="RESOURCE_FAIL",
            name="Resource Failure",
            description="System resource-related failures",
            severity=3,
            probability=0.04,
            mitigation="Implement resource monitoring and scaling"
        )
        
        resource_failure.children = [
            FaultEvent(
                id="MEM_FAIL",
                name="Memory Exhaustion",
                description="System runs out of memory",
                severity=4,
                probability=0.03,
                mitigation="Implement memory monitoring and batch processing"
            ),
            FaultEvent(
                id="GPU_FAIL",
                name="GPU Failure",
                description="GPU processing fails or is unavailable",
                severity=3,
                probability=0.02,
                mitigation="Add CPU fallback and multi-GPU support"
            ),
            FaultEvent(
                id="IO_FAIL",
                name="I/O Failure",
                description="File system or network I/O fails",
                severity=3,
                probability=0.05,
                mitigation="Implement robust I/O error handling and retries"
            )
        ]
        
        # Set up parent-child relationships
        root.children = [data_failure, model_failure, resource_failure]
        for child in root.children:
            child.parent = root
            if child.children:
                for grandchild in child.children:
                    grandchild.parent = child
        
        self.fault_tree = root
        return root
    
    def calculate_risk_metrics(self, event: FaultEvent) -> Dict[str, float]:
        """Calculate risk metrics for a fault event"""
        risk_score = event.severity * event.probability
        
        # Calculate cumulative probability including children
        cumulative_prob = event.probability
        if event.children:
            # Use OR gate probability calculation
            child_probs = [1 - child.probability for child in event.children]
            cumulative_prob = 1 - np.prod(child_probs)
        
        return {
            "risk_score": risk_score,
            "cumulative_probability": cumulative_prob,
            "severity": event.severity,
            "raw_probability": event.probability
        }
    
    def analyze_fault_tree(self) -> Dict[str, Dict]:
        """Analyze the complete fault tree and generate metrics"""
        if not self.fault_tree:
            self.build_fault_tree()
        
        results = {}
        
        def analyze_event(event: FaultEvent):
            metrics = self.calculate_risk_metrics(event)
            results[event.id] = {
                "name": event.name,
                "description": event.description,
                "metrics": metrics,
                "mitigation": event.mitigation
            }
            
            if event.children:
                for child in event.children:
                    analyze_event(child)
        
        analyze_event(self.fault_tree)
        return results
    
    def generate_report(self) -> str:
        """Generate a detailed fault analysis report"""
        analysis = self.analyze_fault_tree()
        
        # Sort events by risk score
        sorted_events = sorted(
            analysis.items(),
            key=lambda x: x[1]["metrics"]["risk_score"],
            reverse=True
        )
        
        # Generate report
        report = ["# Fault Tree Analysis Report", ""]
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("## High Risk Events (Risk Score > 0.5)")
        for event_id, event_data in sorted_events:
            risk_score = event_data["metrics"]["risk_score"]
            if risk_score > 0.5:
                report.extend([
                    f"\n### {event_data['name']} ({event_id})",
                    f"Description: {event_data['description']}",
                    f"Risk Score: {risk_score:.2f}",
                    f"Severity: {event_data['metrics']['severity']}",
                    f"Probability: {event_data['metrics']['raw_probability']:.2f}",
                    f"Mitigation: {event_data['mitigation']}\n"
                ])
        
        report.append("## All Events by Risk Score")
        for event_id, event_data in sorted_events:
            report.extend([
                f"\n### {event_data['name']} ({event_id})",
                f"Description: {event_data['description']}",
                f"Risk Score: {event_data['metrics']['risk_score']:.2f}",
                f"Severity: {event_data['metrics']['severity']}",
                f"Probability: {event_data['metrics']['raw_probability']:.2f}",
                f"Cumulative Probability: {event_data['metrics']['cumulative_probability']:.2f}",
                f"Mitigation: {event_data['mitigation']}\n"
            ])
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / f"fault_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.write_text(report_text)
        
        # Save analysis data as JSON
        json_path = self.output_dir / f"fault_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return report_text
    
    def monitor_system(self, model_metrics: Dict[str, float], 
                      resource_metrics: Dict[str, float]) -> List[str]:
        """Monitor system metrics and identify potential faults"""
        warnings = []
        
        # Check model metrics
        if model_metrics.get('accuracy', 1.0) < 0.9:
            warnings.append("Model accuracy below threshold (0.9)")
        
        if model_metrics.get('loss', 0.0) > 1.0:
            warnings.append("Model loss above threshold (1.0)")
        
        # Check resource metrics
        if resource_metrics.get('memory_usage', 0.0) > 0.9:
            warnings.append("High memory usage (>90%)")
        
        if resource_metrics.get('gpu_memory', 0.0) > 0.9:
            warnings.append("High GPU memory usage (>90%)")
        
        return warnings

def main():
    analyzer = FaultTreeAnalyzer()
    report = analyzer.generate_report()
    print(report)

if __name__ == "__main__":
    main() 