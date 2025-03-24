# Code Complexity Analysis Report

## Summary

- Total files analyzed: 53
- Average cyclomatic complexity: 16.53
- Average Halstead volume: 2174.95
- Average Halstead effort: 31098.21
- Highest cyclomatic complexity: 86 (in handler.py)

## Detailed Metrics

### data_ingestion.py

#### Cyclomatic Complexity: 4
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 45
- Program length: 78
- Volume: 428.36
- Difficulty: 0.88
- Effort: 374.82
- Estimated time to program: 0.01 hours
- Estimated number of bugs: 0.14

### data_processing.py

#### Cyclomatic Complexity: 4
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 56
- Program length: 130
- Volume: 754.96
- Difficulty: 1.17
- Effort: 885.36
- Estimated time to program: 0.01 hours
- Estimated number of bugs: 0.25

### fix_runpod_command.py

#### Cyclomatic Complexity: 5
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 28
- Program length: 46
- Volume: 221.14
- Difficulty: 0.83
- Effort: 184.28
- Estimated time to program: 0.00 hours
- Estimated number of bugs: 0.07

### generate_threat_data.py

#### Cyclomatic Complexity: 13
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 194
- Program length: 490
- Volume: 3723.96
- Difficulty: 14.93
- Effort: 55613.82
- Estimated time to program: 0.86 hours
- Estimated number of bugs: 1.24

### handler.py

#### Cyclomatic Complexity: 86
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 349
- Program length: 1530
- Volume: 12924.04
- Difficulty: 28.07
- Effort: 362776.96
- Estimated time to program: 5.60 hours
- Estimated number of bugs: 4.31

### jamba.py

#### Cyclomatic Complexity: 55
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 335
- Program length: 864
- Volume: 7247.25
- Difficulty: 12.75
- Effort: 92430.27
- Estimated time to program: 1.43 hours
- Estimated number of bugs: 2.42

### jamba/__init__.py

#### Cyclomatic Complexity: 1
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 2
- Program length: 2
- Volume: 2.00
- Difficulty: 0.00
- Effort: 0.00
- Estimated time to program: 0.00 hours
- Estimated number of bugs: 0.00

### jamba/analysis/__init__.py

#### Cyclomatic Complexity: 1
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 7
- Program length: 7
- Volume: 19.65
- Difficulty: 0.00
- Effort: 0.00
- Estimated time to program: 0.00 hours
- Estimated number of bugs: 0.01

### jamba/analysis/analyze_codebase.py

#### Cyclomatic Complexity: 15
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 106
- Program length: 276
- Volume: 1856.91
- Difficulty: 7.86
- Effort: 14595.28
- Estimated time to program: 0.23 hours
- Estimated number of bugs: 0.62

### jamba/analysis/code_complexity.py

#### Cyclomatic Complexity: 20
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 96
- Program length: 373
- Volume: 2456.19
- Difficulty: 13.80
- Effort: 33903.72
- Estimated time to program: 0.52 hours
- Estimated number of bugs: 0.82

### jamba/analysis/fault_tree.py

#### Cyclomatic Complexity: 16
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 173
- Program length: 382
- Volume: 2840.03
- Difficulty: 7.80
- Effort: 22155.64
- Estimated time to program: 0.34 hours
- Estimated number of bugs: 0.95

### jamba/analyze_predictions.py

#### Cyclomatic Complexity: 8
Interpretation:
- Moderate risk - Moderately complex

#### Halstead Metrics
- Program vocabulary: 156
- Program length: 403
- Volume: 2936.02
- Difficulty: 6.34
- Effort: 18617.46
- Estimated time to program: 0.29 hours
- Estimated number of bugs: 0.98

### jamba/cli.py

#### Cyclomatic Complexity: 2
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 38
- Program length: 55
- Volume: 288.64
- Difficulty: 0.73
- Effort: 210.63
- Estimated time to program: 0.00 hours
- Estimated number of bugs: 0.10

### jamba/config.py

#### Cyclomatic Complexity: 5
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 47
- Program length: 97
- Volume: 538.80
- Difficulty: 3.14
- Effort: 1689.86
- Estimated time to program: 0.03 hours
- Estimated number of bugs: 0.18

### jamba/data/datasets.py

#### Cyclomatic Complexity: 9
Interpretation:
- Moderate risk - Moderately complex

#### Halstead Metrics
- Program vocabulary: 50
- Program length: 161
- Volume: 908.66
- Difficulty: 14.29
- Effort: 12980.87
- Estimated time to program: 0.20 hours
- Estimated number of bugs: 0.30

### jamba/data/sample_data.py

#### Cyclomatic Complexity: 3
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 79
- Program length: 204
- Volume: 1285.97
- Difficulty: 5.91
- Effort: 7602.87
- Estimated time to program: 0.12 hours
- Estimated number of bugs: 0.43

### jamba/data_preprocessing.py

#### Cyclomatic Complexity: 13
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 79
- Program length: 239
- Volume: 1506.60
- Difficulty: 7.84
- Effort: 11808.51
- Estimated time to program: 0.18 hours
- Estimated number of bugs: 0.50

### jamba/jamba_model.py

#### Cyclomatic Complexity: 21
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 118
- Program length: 409
- Volume: 2815.00
- Difficulty: 19.84
- Effort: 55852.78
- Estimated time to program: 0.86 hours
- Estimated number of bugs: 0.94

### jamba/model.py

#### Cyclomatic Complexity: 5
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 41
- Program length: 139
- Volume: 744.70
- Difficulty: 3.51
- Effort: 2616.00
- Estimated time to program: 0.04 hours
- Estimated number of bugs: 0.25

### jamba/model/config_validator.py

#### Cyclomatic Complexity: 4
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 46
- Program length: 98
- Volume: 541.31
- Difficulty: 5.55
- Effort: 3003.61
- Estimated time to program: 0.05 hours
- Estimated number of bugs: 0.18

### jamba/model/model_factory.py

#### Cyclomatic Complexity: 5
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 43
- Program length: 106
- Volume: 575.18
- Difficulty: 2.49
- Effort: 1430.95
- Estimated time to program: 0.02 hours
- Estimated number of bugs: 0.19

### jamba/model_config.py

#### Cyclomatic Complexity: 7
Interpretation:
- Moderate risk - Moderately complex

#### Halstead Metrics
- Program vocabulary: 81
- Program length: 203
- Volume: 1286.99
- Difficulty: 6.18
- Effort: 7959.01
- Estimated time to program: 0.12 hours
- Estimated number of bugs: 0.43

### jamba/run_local.py

#### Cyclomatic Complexity: 2
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 54
- Program length: 114
- Volume: 656.06
- Difficulty: 1.07
- Effort: 699.38
- Estimated time to program: 0.01 hours
- Estimated number of bugs: 0.22

### jamba/tests/test_fault_tree.py

#### Cyclomatic Complexity: 6
Interpretation:
- Moderate risk - Moderately complex

#### Halstead Metrics
- Program vocabulary: 90
- Program length: 232
- Volume: 1506.11
- Difficulty: 3.95
- Effort: 5946.54
- Estimated time to program: 0.09 hours
- Estimated number of bugs: 0.50

### jamba/tests/test_regression.py

#### Cyclomatic Complexity: 2
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 88
- Program length: 258
- Volume: 1666.53
- Difficulty: 7.59
- Effort: 12649.59
- Estimated time to program: 0.20 hours
- Estimated number of bugs: 0.56

### jamba/train.py

#### Cyclomatic Complexity: 15
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 174
- Program length: 494
- Volume: 3676.81
- Difficulty: 15.93
- Effort: 58558.34
- Estimated time to program: 0.90 hours
- Estimated number of bugs: 1.23

### jamba/utils/runpod_utils.py

#### Cyclomatic Complexity: 24
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 85
- Program length: 233
- Volume: 1493.39
- Difficulty: 4.08
- Effort: 6091.93
- Estimated time to program: 0.09 hours
- Estimated number of bugs: 0.50

### jamba/version_compatibility.py

#### Cyclomatic Complexity: 9
Interpretation:
- Moderate risk - Moderately complex

#### Halstead Metrics
- Program vocabulary: 52
- Program length: 115
- Volume: 655.55
- Difficulty: 3.34
- Effort: 2187.40
- Estimated time to program: 0.03 hours
- Estimated number of bugs: 0.22

### jamba_model/__init__.py

#### Cyclomatic Complexity: 1
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 3
- Program length: 3
- Volume: 4.75
- Difficulty: 0.00
- Effort: 0.00
- Estimated time to program: 0.00 hours
- Estimated number of bugs: 0.00

### jamba_model/model.py

#### Cyclomatic Complexity: 6
Interpretation:
- Moderate risk - Moderately complex

#### Halstead Metrics
- Program vocabulary: 55
- Program length: 202
- Volume: 1167.83
- Difficulty: 14.22
- Effort: 16605.15
- Estimated time to program: 0.26 hours
- Estimated number of bugs: 0.39

### model_training.py

#### Cyclomatic Complexity: 14
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 202
- Program length: 640
- Volume: 4901.26
- Difficulty: 12.80
- Effort: 62756.28
- Estimated time to program: 0.97 hours
- Estimated number of bugs: 1.63

### predict.py

#### Cyclomatic Complexity: 6
Interpretation:
- Moderate risk - Moderately complex

#### Halstead Metrics
- Program vocabulary: 61
- Program length: 136
- Volume: 806.58
- Difficulty: 3.39
- Effort: 2732.64
- Estimated time to program: 0.04 hours
- Estimated number of bugs: 0.27

### push_to_hub.py

#### Cyclomatic Complexity: 9
Interpretation:
- Moderate risk - Moderately complex

#### Halstead Metrics
- Program vocabulary: 71
- Program length: 132
- Volume: 811.77
- Difficulty: 2.82
- Effort: 2292.05
- Estimated time to program: 0.04 hours
- Estimated number of bugs: 0.27

### run_simplified.py

#### Cyclomatic Complexity: 3
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 132
- Program length: 260
- Volume: 1831.54
- Difficulty: 1.98
- Effort: 3634.91
- Estimated time to program: 0.06 hours
- Estimated number of bugs: 0.61

### run_training.py

#### Cyclomatic Complexity: 5
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 74
- Program length: 134
- Volume: 832.07
- Difficulty: 1.83
- Effort: 1525.46
- Estimated time to program: 0.02 hours
- Estimated number of bugs: 0.28

### runpod_client.py

#### Cyclomatic Complexity: 41
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 203
- Program length: 560
- Volume: 4292.59
- Difficulty: 18.17
- Effort: 77978.25
- Estimated time to program: 1.20 hours
- Estimated number of bugs: 1.43

### runpod_health_check.py

#### Cyclomatic Complexity: 32
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 123
- Program length: 349
- Volume: 2422.94
- Difficulty: 12.67
- Effort: 30701.17
- Estimated time to program: 0.47 hours
- Estimated number of bugs: 0.81

### runpod_startup.py

#### Cyclomatic Complexity: 32
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 135
- Program length: 449
- Volume: 3177.49
- Difficulty: 11.76
- Effort: 37360.33
- Estimated time to program: 0.58 hours
- Estimated number of bugs: 1.06

### runpod_verify.py

#### Cyclomatic Complexity: 21
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 117
- Program length: 250
- Volume: 1717.59
- Difficulty: 5.38
- Effort: 9239.72
- Estimated time to program: 0.14 hours
- Estimated number of bugs: 0.57

### test_gpu.py

#### Cyclomatic Complexity: 5
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 22
- Program length: 45
- Volume: 200.67
- Difficulty: 1.05
- Effort: 210.23
- Estimated time to program: 0.00 hours
- Estimated number of bugs: 0.07

### test_runpod_connection.py

#### Cyclomatic Complexity: 6
Interpretation:
- Moderate risk - Moderately complex

#### Halstead Metrics
- Program vocabulary: 53
- Program length: 95
- Volume: 544.15
- Difficulty: 0.89
- Effort: 486.60
- Estimated time to program: 0.01 hours
- Estimated number of bugs: 0.18

### train_jamba.py

#### Cyclomatic Complexity: 80
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 362
- Program length: 1053
- Volume: 8950.34
- Difficulty: 23.95
- Effort: 214341.13
- Estimated time to program: 3.31 hours
- Estimated number of bugs: 2.98

### train_local.py

#### Cyclomatic Complexity: 2
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 100
- Program length: 252
- Volume: 1674.25
- Difficulty: 7.50
- Effort: 12556.89
- Estimated time to program: 0.19 hours
- Estimated number of bugs: 0.56

### train_local_sample.py

#### Cyclomatic Complexity: 14
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 127
- Program length: 274
- Volume: 1914.90
- Difficulty: 6.60
- Effort: 12628.84
- Estimated time to program: 0.19 hours
- Estimated number of bugs: 0.64

### train_runpod.py

#### Cyclomatic Complexity: 37
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 278
- Program length: 814
- Volume: 6608.82
- Difficulty: 20.52
- Effort: 135630.97
- Estimated time to program: 2.09 hours
- Estimated number of bugs: 2.20

### train_with_intel.py

#### Cyclomatic Complexity: 6
Interpretation:
- Moderate risk - Moderately complex

#### Halstead Metrics
- Program vocabulary: 63
- Program length: 120
- Volume: 717.27
- Difficulty: 3.80
- Effort: 2723.21
- Estimated time to program: 0.04 hours
- Estimated number of bugs: 0.24

### train_with_runpod.py

#### Cyclomatic Complexity: 4
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 55
- Program length: 95
- Volume: 549.23
- Difficulty: 1.74
- Effort: 953.38
- Estimated time to program: 0.01 hours
- Estimated number of bugs: 0.18

### utils/__init__.py

#### Cyclomatic Complexity: 1
Interpretation:
- Low risk - Simple code

#### Halstead Metrics
- Program vocabulary: 6
- Program length: 6
- Volume: 15.51
- Difficulty: 0.00
- Effort: 0.00
- Estimated time to program: 0.00 hours
- Estimated number of bugs: 0.01

### utils/cli.py

#### Cyclomatic Complexity: 38
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 142
- Program length: 386
- Volume: 2759.80
- Difficulty: 9.05
- Effort: 24971.10
- Estimated time to program: 0.39 hours
- Estimated number of bugs: 0.92

### utils/environment.py

#### Cyclomatic Complexity: 20
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 96
- Program length: 252
- Volume: 1659.41
- Difficulty: 6.70
- Effort: 11123.52
- Estimated time to program: 0.17 hours
- Estimated number of bugs: 0.55

### utils/model_testing.py

#### Cyclomatic Complexity: 24
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 119
- Program length: 381
- Volume: 2626.93
- Difficulty: 11.19
- Effort: 29388.73
- Estimated time to program: 0.45 hours
- Estimated number of bugs: 0.88

### utils/shell.py

#### Cyclomatic Complexity: 61
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 140
- Program length: 412
- Volume: 2937.26
- Difficulty: 8.44
- Effort: 24791.39
- Estimated time to program: 0.38 hours
- Estimated number of bugs: 0.98

### utils/validation.py

#### Cyclomatic Complexity: 48
Interpretation:
- High risk - Complex code, consider refactoring

#### Halstead Metrics
- Program vocabulary: 270
- Program length: 816
- Volume: 6590.68
- Difficulty: 21.36
- Effort: 140747.09
- Estimated time to program: 2.17 hours
- Estimated number of bugs: 2.20
