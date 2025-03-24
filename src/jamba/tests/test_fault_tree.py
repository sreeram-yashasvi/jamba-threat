#!/usr/bin/env python3
import unittest
import tempfile
import shutil
from pathlib import Path
import json

from jamba.analysis.fault_tree import FaultTreeAnalyzer, FaultEvent

class TestFaultTreeAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp()
        cls.analyzer = FaultTreeAnalyzer(output_dir=cls.test_dir)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)
    
    def test_fault_tree_creation(self):
        """Test creation of fault tree structure"""
        root = self.analyzer.build_fault_tree()
        
        # Test root node
        self.assertEqual(root.id, "SYS_FAIL")
        self.assertEqual(len(root.children), 3)
        
        # Test children nodes
        child_ids = {child.id for child in root.children}
        expected_ids = {"DATA_FAIL", "MODEL_FAIL", "RESOURCE_FAIL"}
        self.assertEqual(child_ids, expected_ids)
        
        # Test parent-child relationships
        for child in root.children:
            self.assertEqual(child.parent, root)
            if child.children:
                for grandchild in child.children:
                    self.assertEqual(grandchild.parent, child)
    
    def test_risk_metrics_calculation(self):
        """Test calculation of risk metrics"""
        event = FaultEvent(
            id="TEST",
            name="Test Event",
            description="Test description",
            severity=4,
            probability=0.2,
            mitigation="Test mitigation"
        )
        
        metrics = self.analyzer.calculate_risk_metrics(event)
        
        self.assertEqual(metrics["severity"], 4)
        self.assertEqual(metrics["raw_probability"], 0.2)
        self.assertEqual(metrics["risk_score"], 0.8)  # 4 * 0.2
        self.assertEqual(metrics["cumulative_probability"], 0.2)
        
        # Test with children
        event.children = [
            FaultEvent(
                id="CHILD1",
                name="Child 1",
                description="Child 1",
                severity=3,
                probability=0.1,
                mitigation="Test"
            ),
            FaultEvent(
                id="CHILD2",
                name="Child 2",
                description="Child 2",
                severity=2,
                probability=0.15,
                mitigation="Test"
            )
        ]
        
        metrics = self.analyzer.calculate_risk_metrics(event)
        expected_cumulative_prob = 1 - (0.9 * 0.85)  # 1 - ((1-0.1) * (1-0.15))
        self.assertAlmostEqual(metrics["cumulative_probability"], expected_cumulative_prob)
    
    def test_fault_tree_analysis(self):
        """Test complete fault tree analysis"""
        analysis = self.analyzer.analyze_fault_tree()
        
        # Check all events are analyzed
        self.assertIn("SYS_FAIL", analysis)
        self.assertIn("DATA_FAIL", analysis)
        self.assertIn("MODEL_FAIL", analysis)
        self.assertIn("RESOURCE_FAIL", analysis)
        
        # Check metrics are calculated
        for event_data in analysis.values():
            self.assertIn("metrics", event_data)
            metrics = event_data["metrics"]
            self.assertIn("risk_score", metrics)
            self.assertIn("cumulative_probability", metrics)
            self.assertIn("severity", metrics)
            self.assertIn("raw_probability", metrics)
    
    def test_report_generation(self):
        """Test report generation"""
        report = self.analyzer.generate_report()
        
        # Check report content
        self.assertIn("# Fault Tree Analysis Report", report)
        self.assertIn("## High Risk Events", report)
        self.assertIn("## All Events by Risk Score", report)
        
        # Check report files are created
        report_files = list(Path(self.test_dir).glob("fault_analysis_*.md"))
        json_files = list(Path(self.test_dir).glob("fault_analysis_*.json"))
        
        self.assertEqual(len(report_files), 1)
        self.assertEqual(len(json_files), 1)
        
        # Check JSON content
        with open(json_files[0]) as f:
            data = json.load(f)
            self.assertIn("SYS_FAIL", data)
            self.assertIn("metrics", data["SYS_FAIL"])
    
    def test_system_monitoring(self):
        """Test system monitoring functionality"""
        # Test with metrics below thresholds
        model_metrics = {"accuracy": 0.95, "loss": 0.5}
        resource_metrics = {"memory_usage": 0.7, "gpu_memory": 0.6}
        
        warnings = self.analyzer.monitor_system(model_metrics, resource_metrics)
        self.assertEqual(len(warnings), 0)
        
        # Test with metrics above thresholds
        model_metrics = {"accuracy": 0.85, "loss": 1.2}
        resource_metrics = {"memory_usage": 0.95, "gpu_memory": 0.92}
        
        warnings = self.analyzer.monitor_system(model_metrics, resource_metrics)
        self.assertEqual(len(warnings), 4)
        self.assertIn("Model accuracy below threshold (0.9)", warnings)
        self.assertIn("Model loss above threshold (1.0)", warnings)
        self.assertIn("High memory usage (>90%)", warnings)
        self.assertIn("High GPU memory usage (>90%)", warnings)

if __name__ == "__main__":
    unittest.main() 