import ast
import math
from typing import Dict, List, Set, Tuple

class HalsteadMetrics:
    """Calculate Halstead complexity metrics."""
    
    def __init__(self):
        self.operators: Set[str] = set()
        self.operands: Set[str] = set()
        self.operator_counts: Dict[str, int] = {}
        self.operand_counts: Dict[str, int] = {}
    
    def visit_node(self, node: ast.AST) -> None:
        """Visit an AST node and collect operators and operands."""
        # Operators from different node types
        if isinstance(node, ast.BinOp):
            op = self._get_operator_symbol(node.op)
            self.operators.add(op)
            self.operator_counts[op] = self.operator_counts.get(op, 0) + 1
            
        elif isinstance(node, ast.UnaryOp):
            op = self._get_operator_symbol(node.op)
            self.operators.add(op)
            self.operator_counts[op] = self.operator_counts.get(op, 0) + 1
            
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                op_symbol = self._get_operator_symbol(op)
                self.operators.add(op_symbol)
                self.operator_counts[op_symbol] = self.operator_counts.get(op_symbol, 0) + 1
                
        elif isinstance(node, ast.BoolOp):
            op = self._get_operator_symbol(node.op)
            self.operators.add(op)
            self.operator_counts[op] = self.operator_counts.get(op, 0) + 1
            
        # Operands from different node types
        elif isinstance(node, ast.Name):
            self.operands.add(node.id)
            self.operand_counts[node.id] = self.operand_counts.get(node.id, 0) + 1
            
        elif isinstance(node, ast.Num):
            operand = str(node.n)
            self.operands.add(operand)
            self.operand_counts[operand] = self.operand_counts.get(operand, 0) + 1
            
        elif isinstance(node, ast.Str):
            operand = node.s
            self.operands.add(operand)
            self.operand_counts[operand] = self.operand_counts.get(operand, 0) + 1
        
        # Recursively visit child nodes
        for child in ast.iter_child_nodes(node):
            self.visit_node(child)
    
    def _get_operator_symbol(self, op: ast.AST) -> str:
        """Convert AST operator to string symbol."""
        op_map = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '//',
            ast.Mod: '%',
            ast.Pow: '**',
            ast.LShift: '<<',
            ast.RShift: '>>',
            ast.BitOr: '|',
            ast.BitXor: '^',
            ast.BitAnd: '&',
            ast.MatMult: '@',
            ast.And: 'and',
            ast.Or: 'or',
            ast.Not: 'not',
            ast.Eq: '==',
            ast.NotEq: '!=',
            ast.Lt: '<',
            ast.LtE: '<=',
            ast.Gt: '>',
            ast.GtE: '>=',
            ast.Is: 'is',
            ast.IsNot: 'is not',
            ast.In: 'in',
            ast.NotIn: 'not in',
            ast.UAdd: '+',
            ast.USub: '-',
            ast.Invert: '~'
        }
        return op_map.get(type(op), str(type(op)))
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate Halstead metrics."""
        n1 = len(self.operators)  # Number of unique operators
        n2 = len(self.operands)   # Number of unique operands
        N1 = sum(self.operator_counts.values())  # Total operators
        N2 = sum(self.operand_counts.values())   # Total operands
        
        # Calculate basic metrics
        vocabulary = n1 + n2
        length = N1 + N2
        
        # Avoid division by zero and log of zero
        if vocabulary <= 0 or length <= 0:
            return {
                "vocabulary": vocabulary,
                "length": length,
                "volume": 0,
                "difficulty": 0,
                "effort": 0,
                "time": 0,
                "bugs": 0
            }
        
        # Calculate derived metrics
        volume = length * math.log2(vocabulary)
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = difficulty * volume
        time = effort / 18  # Estimated time in seconds
        bugs = volume / 3000  # Estimated number of bugs
        
        return {
            "vocabulary": vocabulary,
            "length": length,
            "volume": volume,
            "difficulty": difficulty,
            "effort": effort,
            "time": time,
            "bugs": bugs
        }

class CyclomaticComplexity:
    """Calculate cyclomatic complexity."""
    
    def __init__(self):
        self.complexity = 1  # Start with 1 for the entry point
    
    def visit_node(self, node: ast.AST) -> None:
        """Visit an AST node and count decision points."""
        # Count control flow statements
        if isinstance(node, (ast.If, ast.While, ast.For)):
            self.complexity += 1
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And) or isinstance(node.op, ast.Or):
                self.complexity += len(node.values) - 1
        elif isinstance(node, ast.Try):
            self.complexity += len(node.handlers)  # Count except blocks
        
        # Recursively visit child nodes
        for child in ast.iter_child_nodes(node):
            self.visit_node(child)

def analyze_code_complexity(code: str) -> Dict[str, Dict[str, float]]:
    """Analyze both Halstead and cyclomatic complexity of code."""
    try:
        tree = ast.parse(code)
        
        # Calculate Halstead metrics
        halstead = HalsteadMetrics()
        halstead.visit_node(tree)
        halstead_metrics = halstead.calculate_metrics()
        
        # Calculate cyclomatic complexity
        cyclomatic = CyclomaticComplexity()
        cyclomatic.visit_node(tree)
        
        return {
            "halstead": halstead_metrics,
            "cyclomatic": {"complexity": cyclomatic.complexity}
        }
    except Exception as e:
        return {
            "error": f"Failed to analyze code: {str(e)}"
        }

def analyze_file_complexity(file_path: str) -> Dict[str, Dict[str, float]]:
    """Analyze complexity metrics for a Python file."""
    try:
        with open(file_path, 'r') as f:
            code = f.read()
        return analyze_code_complexity(code)
    except Exception as e:
        return {
            "error": f"Failed to analyze file {file_path}: {str(e)}"
        } 