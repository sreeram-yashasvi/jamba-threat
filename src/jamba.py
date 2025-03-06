#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jamba Threat Detection - Main CLI Interface

This script provides a command-line interface to the Jamba Threat Detection system.
It allows users to train models, make predictions, verify system health, and more.
"""

import os
import sys
import logging
import argparse
import json
import time
from pathlib import Path

# Add the parent directory to the path to import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

try:
    # Import utility modules
    from utils.environment import initialize_environment, get_model_dir, get_data_dir, get_log_dir
    from utils.validation import check_environment_variables, check_cuda_availability
    from utils.validation import check_model_imports, test_model_initialization, test_model_serialization
    from utils.validation import check_runpod_endpoint, run_health_check
    
    # Import handler functionality
    from handler import handler as jamba_handler
    
    utilities_imported = True
except ImportError as e:
    utilities_imported = False
    logging.error(f"Error importing utilities: {e}")
    # Try alternative import paths
    try:
        sys.path.insert(0, os.path.join(project_root, 'src'))
        from utils.environment import initialize_environment, get_model_dir, get_data_dir, get_log_dir
        from utils.validation import check_environment_variables, check_cuda_availability
        from utils.validation import check_model_imports, test_model_initialization, test_model_serialization
        from utils.validation import check_runpod_endpoint, run_health_check
        from handler import handler as jamba_handler
        utilities_imported = True
    except ImportError as e:
        logging.error(f"Failed to import utilities even after path adjustment: {e}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Jamba Threat Detection CLI")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify system health and compatibility')
    verify_parser.add_argument('--skip-endpoint', action='store_true', help='Skip RunPod endpoint checks')
    verify_parser.add_argument('--api-key', help='RunPod API key (overrides environment variable)')
    verify_parser.add_argument('--endpoint-id', help='RunPod endpoint ID (overrides environment variable)')
    verify_parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data-path', required=True, help='Path to training data (CSV or JSON)')
    train_parser.add_argument('--input-dim', type=int, default=512, help='Input dimension')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--model-name', help='Output model name')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions using a trained model')
    predict_parser.add_argument('--model-path', help='Path to model file')
    predict_parser.add_argument('--input-file', required=True, help='File containing input data (JSON or CSV)')
    predict_parser.add_argument('--output-file', help='Output file for predictions (JSON)')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Run system health check')
    health_parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    # Environment command
    env_parser = subparsers.add_parser('env', help='Check and display environment information')
    env_parser.add_argument('--json', action='store_true', help='Output results as JSON')
    
    # Service command
    service_parser = subparsers.add_parser('service', help='Run as a service')
    service_parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    service_parser.add_argument('--host', default='0.0.0.0', help='Host address to bind to')
    
    return parser.parse_args()


def run_verify_command(args):
    """Run the verify command"""
    start_time = time.time()
    
    logging.info("Running system verification...")
    
    # Initialize environment
    initialize_environment()
    
    # Results dictionary
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": {}
    }
    
    # Check environment variables
    env_result = check_environment_variables()
    results["checks"]["environment"] = env_result
    if env_result["status"] == "pass":
        logging.info("✓ Environment variables check passed")
    else:
        logging.warning("⚠️ Environment variables check failed")
        for var, status in env_result["variables"].items():
            if not status:
                logging.warning(f"  - Missing: {var}")
    
    # Check CUDA
    cuda_result = check_cuda_availability()
    results["checks"]["cuda"] = cuda_result
    if cuda_result["available"]:
        logging.info(f"✓ CUDA is available (version {cuda_result['version']})")
        logging.info(f"  - Device count: {cuda_result['device_count']}")
        if cuda_result['device_count'] > 0:
            logging.info(f"  - Current device: {cuda_result['current_device']['name']}")
            logging.info(f"  - Memory: {cuda_result['current_device']['memory_total']/1024/1024:.1f} MB")
    else:
        logging.warning("⚠️ CUDA is not available, using CPU")
    
    # Check model imports
    import_result = check_model_imports()
    results["checks"]["imports"] = import_result
    if import_result["success"]:
        logging.info("✓ Model imports successful")
    else:
        logging.error("✗ Model imports failed")
        logging.error(f"  - Error: {import_result['error']}")
    
    # Check model initialization
    init_result = test_model_initialization()
    results["checks"]["model_init"] = init_result
    if init_result["success"]:
        logging.info("✓ Model initialization successful")
        logging.info(f"  - Model architecture: {init_result['architecture']}")
        logging.info(f"  - Output shape: {init_result['output_shape']}")
    else:
        logging.error("✗ Model initialization failed")
        logging.error(f"  - Error: {init_result['error']}")
    
    # Check model serialization
    serialization_result = test_model_serialization()
    results["checks"]["serialization"] = serialization_result
    if serialization_result["success"]:
        logging.info("✓ Model serialization test passed")
    else:
        logging.error("✗ Model serialization test failed")
        logging.error(f"  - Error: {serialization_result['error']}")
    
    # Check RunPod endpoint if not skipped
    if not args.skip_endpoint:
        api_key = args.api_key or os.environ.get('RUNPOD_API_KEY')
        endpoint_id = args.endpoint_id or os.environ.get('RUNPOD_ENDPOINT_ID')
        
        if api_key and endpoint_id:
            logging.info(f"Checking RunPod endpoint {endpoint_id}...")
            endpoint_result = check_runpod_endpoint(api_key, endpoint_id)
            results["checks"]["endpoint"] = endpoint_result
            
            if endpoint_result["reachable"]:
                logging.info("✓ RunPod endpoint is reachable")
                
                if endpoint_result["health_check_success"]:
                    logging.info("✓ Endpoint health check passed")
                else:
                    logging.error("✗ Endpoint health check failed")
                    if "error" in endpoint_result:
                        logging.error(f"  - Error: {endpoint_result['error']}")
            else:
                logging.error("✗ RunPod endpoint is not reachable")
                if "error" in endpoint_result:
                    logging.error(f"  - Error: {endpoint_result['error']}")
        else:
            logging.warning("⚠️ Skipping RunPod endpoint check - API key or endpoint ID not provided")
            results["checks"]["endpoint"] = {"skipped": True, "reason": "API key or endpoint ID not provided"}
    else:
        logging.info("Skipping RunPod endpoint check as requested")
        results["checks"]["endpoint"] = {"skipped": True, "reason": "User requested to skip"}
    
    # Summarize results
    all_passed = all(
        check.get("success", check.get("available", check.get("status", "") == "pass"))
        for name, check in results["checks"].items()
        if not check.get("skipped", False)
    )
    
    results["execution_time"] = f"{time.time() - start_time:.2f}s"
    results["all_passed"] = all_passed
    
    logging.info("=" * 60)
    logging.info(f"Verification completed in {results['execution_time']}")
    if all_passed:
        logging.info("✓ All checks passed successfully!")
    else:
        logging.warning("⚠️ Some checks failed - see details above")
    
    if args.json:
        print(json.dumps(results, indent=2))
    
    return results


def run_train_command(args):
    """Run the train command"""
    logging.info(f"Training model with data from {args.data_path}")
    
    # Prepare parameters
    params = {
        "input_dim": args.input_dim,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate
    }
    
    if args.model_name:
        params["model_name"] = args.model_name
    
    # Create event for handler
    event = {
        "operation": "train",
        "data_path": args.data_path,
        "params": params
    }
    
    # Call handler
    result = jamba_handler(event)
    
    if result.get("success", False):
        logging.info(f"Training completed successfully!")
        logging.info(f"Model saved to: {result['model_path']}")
        
        # Print final metrics
        if "history" in result:
            history = result["history"]
            epochs = len(history["train_loss"])
            final_train_loss = history["train_loss"][-1]
            final_val_loss = history["val_loss"][-1]
            final_train_acc = history["train_acc"][-1]
            final_val_acc = history["val_acc"][-1]
            
            logging.info(f"Final metrics after {epochs} epochs:")
            logging.info(f"  Train loss: {final_train_loss:.4f}")
            logging.info(f"  Validation loss: {final_val_loss:.4f}")
            logging.info(f"  Train accuracy: {final_train_acc:.2f}%")
            logging.info(f"  Validation accuracy: {final_val_acc:.2f}%")
    else:
        logging.error("Training failed!")
        if "error" in result:
            logging.error(f"Error: {result['error']}")
    
    return result


def run_predict_command(args):
    """Run the predict command"""
    import pandas as pd
    import json
    
    # Determine model path
    model_path = args.model_path
    if not model_path:
        model_dir = get_model_dir()
        default_model = os.path.join(model_dir, "jamba_model.pt")
        if os.path.exists(default_model):
            model_path = default_model
            logging.info(f"Using default model at {model_path}")
        else:
            # Try to find the newest model file
            model_files = list(Path(model_dir).glob("*.pt"))
            if model_files:
                newest_model = max(model_files, key=lambda p: p.stat().st_mtime)
                model_path = str(newest_model)
                logging.info(f"Using latest model at {model_path}")
            else:
                logging.error("No model file found and no --model-path specified")
                return {"error": "No model file found"}
    
    # Load input data
    logging.info(f"Loading input data from {args.input_file}")
    try:
        if args.input_file.endswith('.csv'):
            df = pd.read_csv(args.input_file)
            # Convert DataFrame to feature list
            features = df.values.tolist()
        elif args.input_file.endswith('.json'):
            with open(args.input_file, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                features = data
            elif isinstance(data, dict) and 'features' in data:
                features = data['features']
            else:
                features = [list(data.values())]
        else:
            logging.error(f"Unsupported input file format: {args.input_file}")
            return {"error": f"Unsupported input file format: {args.input_file}"}
        
        logging.info(f"Loaded {len(features)} input samples")
        
        # Create event for handler
        event = {
            "operation": "predict",
            "model_path": model_path,
            "input_data": {
                "features": features
            }
        }
        
        # Call handler
        result = jamba_handler(event)
        
        if "error" not in result:
            logging.info("Prediction completed successfully!")
            
            # Save to output file if specified
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                logging.info(f"Results saved to {args.output_file}")
            else:
                # Print results
                print(json.dumps(result, indent=2))
        else:
            logging.error(f"Prediction failed: {result['error']}")
        
        return result
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {"error": str(e)}


def run_health_command(args):
    """Run the health command"""
    logging.info("Running health check...")
    
    result = run_health_check()
    
    if result["status"] == "healthy":
        logging.info("✓ System is healthy")
    else:
        logging.error(f"✗ System health check failed: {result.get('reason', 'Unknown error')}")
    
    if args.json:
        print(json.dumps(result, indent=2))
    
    return result


def run_env_command(args):
    """Run the environment command"""
    logging.info("Checking environment...")
    
    # Initialize environment
    initialize_environment()
    
    # Get directories
    model_dir = get_model_dir()
    data_dir = get_data_dir()
    log_dir = get_log_dir()
    
    # Check environment variables
    env_result = check_environment_variables()
    
    # Prepare result
    result = {
        "directories": {
            "model_dir": model_dir,
            "data_dir": data_dir,
            "log_dir": log_dir
        },
        "environment_variables": env_result["variables"],
        "python_version": sys.version,
        "system_info": {
            "platform": sys.platform,
            "python_path": sys.executable
        }
    }
    
    # Check CUDA
    cuda_result = check_cuda_availability()
    result["cuda"] = cuda_result
    
    # Display information
    logging.info("Environment Information:")
    logging.info(f"  Model Directory: {model_dir}")
    logging.info(f"  Data Directory: {data_dir}")
    logging.info(f"  Log Directory: {log_dir}")
    
    logging.info("Environment Variables:")
    for var, status in env_result["variables"].items():
        log_func = logging.info if status else logging.warning
        status_str = "✓ Set" if status else "✗ Not set"
        log_func(f"  {var}: {status_str}")
    
    logging.info(f"Python Version: {sys.version}")
    
    if cuda_result["available"]:
        logging.info(f"CUDA: Available (version {cuda_result['version']})")
        logging.info(f"  Device Count: {cuda_result['device_count']}")
        if cuda_result['device_count'] > 0:
            logging.info(f"  Current Device: {cuda_result['current_device']['name']}")
    else:
        logging.info("CUDA: Not available")
    
    if args.json:
        print(json.dumps(result, indent=2))
    
    return result


def run_service_command(args):
    """Run as a service"""
    logging.info(f"Starting Jamba Threat Detection service on {args.host}:{args.port}")
    
    try:
        import runpod
        runpod.serverless.start({"handler": jamba_handler})
        logging.info("RunPod serverless service started")
    except ImportError:
        logging.error("RunPod module not found. Cannot start service.")
        logging.info("Attempting to start HTTP service instead...")
        
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import json
            
            class JambaHandler(BaseHTTPRequestHandler):
                def do_POST(self):
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    
                    try:
                        event = json.loads(post_data.decode('utf-8'))
                        result = jamba_handler(event)
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        
                        self.wfile.write(json.dumps(result).encode('utf-8'))
                    except Exception as e:
                        self.send_response(500)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        
                        error = {"error": str(e)}
                        self.wfile.write(json.dumps(error).encode('utf-8'))
                
                def do_GET(self):
                    # Simple health check
                    if self.path == '/health':
                        result = run_health_check()
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        
                        self.wfile.write(json.dumps(result).encode('utf-8'))
                    else:
                        self.send_response(404)
                        self.end_headers()
            
            server = HTTPServer((args.host, args.port), JambaHandler)
            logging.info(f"HTTP service started on {args.host}:{args.port}")
            logging.info("Press Ctrl+C to stop the server")
            server.serve_forever()
            
        except Exception as e:
            logging.error(f"Failed to start service: {e}")
            return {"error": str(e)}
    
    return {"status": "service_started"}


def main():
    """Main entry point"""
    args = parse_args()
    
    # Initialize environment
    if utilities_imported:
        initialize_environment()
    
    # Execute the selected command
    if args.command == 'verify':
        return run_verify_command(args)
    elif args.command == 'train':
        return run_train_command(args)
    elif args.command == 'predict':
        return run_predict_command(args)
    elif args.command == 'health':
        return run_health_command(args)
    elif args.command == 'env':
        return run_env_command(args)
    elif args.command == 'service':
        return run_service_command(args)
    else:
        logging.error(f"Unknown command: {args.command}")
        return {"error": f"Unknown command: {args.command}"}


if __name__ == "__main__":
    main() 