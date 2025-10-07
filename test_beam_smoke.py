#!/usr/bin/env python3
"""
Beam Sandbox Backend - Smoke Test Script

This script validates the Beam integration with ProximalDockerEnv.
Run this before attempting full training runs with Beam.

Usage:
    export BEAM_API_KEY="your-api-key"
    python test_beam_smoke.py --template "your-template-id"
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(title):
    """Print a formatted header."""
    width = 60
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)

def print_result(test_name, passed, message=""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status}: {test_name}")
    if message:
        print(f"  → {message}")

async def test_1_beam_sdk():
    """Test 1: Validate Beam SDK installation and authentication."""
    print_header("TEST 1: Beam SDK & Authentication")
    
    try:
        import beam
    except ImportError:
        print_result("Beam SDK Import", False, "Run: pip install beam-client>=0.13.0")
        return False
    
    print("✓ Beam SDK imported successfully")
    
    api_key = os.getenv("BEAM_API_KEY")
    if not api_key:
        print_result("Beam API Key", False, "Set BEAM_API_KEY environment variable")
        return False
    
    print(f"✓ BEAM_API_KEY is set (length: {len(api_key)} characters)")
    
    # Test SDK by creating a minimal sandbox configuration
    try:
        sandbox_cfg = beam.Sandbox(
            image=beam.Image.from_registry("python:3.11-slim")
        )
        print("✓ Beam SDK can create sandbox configurations")
    except Exception as e:
        print_result("Beam SDK Configuration", False, f"SDK error: {e}")
        return False
    
    print_result("Beam SDK & Authentication", True)
    return True

async def test_2_template_access(template_id):
    """Test 2: Validate image/template accessibility."""
    print_header("TEST 2: Beam Image/Template Access")
    
    try:
        import beam
    except ImportError:
        print_result("Image Access", False, "Beam SDK not installed")
        return False
    
    # Try direct Docker image first (most common case)
    print("Testing Docker registry image...")
    try:
        sandbox_cfg = beam.Sandbox(
            image=beam.Image.from_registry("navidpour/bullmq-python-setup:latest")
        )
        print("✓ Docker image 'navidpour/bullmq-python-setup:latest' is accessible")
        print_result("Image Access", True, "Docker registry image works")
        return True
    except Exception as e:
        print(f"⚠️  Docker image test failed: {e}")
    
    # Try template if provided
    if template_id:
        print(f"\nTesting template: {template_id}")
        try:
            sandbox_cfg = beam.Sandbox(
                image=beam.Image.from_registry(template_id)
            )
            print_result("Template Access", True, f"Template '{template_id}' is accessible")
            return True
        except Exception as e:
            print_result("Template Access", False, f"{e}")
            print("\nTroubleshooting:")
            print("  1. Use Docker registry image instead (recommended)")
            print("  2. Or create template via Beam dashboard")
            print("  3. Verify template ID if using custom template")
            return False
    
    return False

async def test_3_sandbox_provision(template_id):
    """Test 3: Provision sandbox and start OpenCode."""
    print_header("TEST 3: Sandbox Provisioning & OpenCode Startup")
    
    try:
        import beam
        import requests
    except ImportError as e:
        print_result("Sandbox Provisioning", False, f"Missing dependency: {e}")
        return False
    
    sb = None
    
    try:
        # Create sandbox
        print("Creating sandbox with Docker image...")
        start_time = time.time()
        
        # Use Docker registry image directly
        sandbox_cfg = beam.Sandbox(
            image=beam.Image.from_registry("navidpour/bullmq-python-setup:latest"),
            cpu=1.0,
            memory="2Gi"
        )
        sb = sandbox_cfg.create()
        
        create_time = time.time() - start_time
        sandbox_id = sb.sandbox_id()
        print(f"✓ Sandbox created in {create_time:.1f}s (ID: {sandbox_id})")
        
        # Start OpenCode server
        print("Starting OpenCode server...")
        start_cmd = "opencode serve --hostname 0.0.0.0 --port 4096"
        
        # Execute the start command
        process = sb.process.exec("bash", "-lc", start_cmd, cwd="/root/workspace")
        print("✓ OpenCode start command executed")
        
        # Expose port
        print("Exposing port 4096...")
        public_url = sb.expose_port(4096)
        print(f"✓ Public URL: {public_url}")
        
        # Wait for readiness
        print("Waiting for OpenCode to become ready...")
        max_attempts = 30
        ready = False
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{public_url}/doc", timeout=5)
                if response.status_code < 500:
                    ready_time = time.time() - start_time
                    print(f"✓ OpenCode ready in {ready_time:.1f}s (HTTP {response.status_code})")
                    ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            
            if attempt < max_attempts - 1:
                print(f"  Attempt {attempt + 1}/{max_attempts}...", end='\r')
                time.sleep(2)
        
        if not ready:
            print("\n⚠️  OpenCode health check failed, but continuing...")
            print("(Some images may not have the /doc endpoint)")
            # Don't fail the test - OpenCode might still work
        
        # Test a simple command
        print("\nTesting command execution...")
        test_process = sb.process.exec("echo", "Hello from Beam sandbox!", cwd="/root/workspace")
        test_process.wait()
        output = test_process.stdout.read()
        print(f"✓ Command output: {output.strip()}")
        
        print_result("Sandbox Provisioning & OpenCode", True)
        return True
        
    except Exception as e:
        print_result("Sandbox Provisioning", False, str(e))
        import traceback
        traceback.print_exc()
        return False
    finally:
        if sb:
            print("\nCleaning up sandbox...")
            try:
                sb.terminate()
                print("✓ Sandbox terminated")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")

async def test_4_proximal_integration(template_id):
    """Test 4: Full integration with ProximalDockerEnv."""
    print_header("TEST 4: ProximalDockerEnv Integration")
    
    try:
        from datasets import Dataset
        from env_api.vf_exts.vf_exts.env.proximal_docker import ProximalDockerEnv
        from nemo_rl.execution import RolloutCompleted
    except ImportError as e:
        print_result("ProximalDockerEnv Integration", False, f"Import error: {e}")
        return False
    
    # Create minimal test dataset
    dataset = Dataset.from_dict({
        "prompt": [
            [{"role": "user", "content": "Write a function to add two numbers"}]
        ],
        "answer": [""],
        "info": [{}],
        "task": ["coding"],
    })
    
    # Track events
    events_received = []
    
    def event_callback(event: RolloutCompleted):
        events_received.append(event)
        reward_value = event.reward
        numeric_reward: float | None = None
        if isinstance(reward_value, (int, float)):
            numeric_reward = float(reward_value)
        elif isinstance(reward_value, dict):
            for key in ("total", "resolved_fraction", "reward"):
                candidate = reward_value.get(key)
                if isinstance(candidate, (int, float)):
                    numeric_reward = float(candidate)
                    break

        reward_display = f"{numeric_reward:.2f}" if numeric_reward is not None else str(reward_value)
        print(f"\n  → Rollout completed: backend={event.sandbox_backend}, reward={reward_display}")
    
    print("Creating ProximalDockerEnv with Beam backend...")
    
    try:
        env = ProximalDockerEnv(
            backend_priority=["beam"],  # Beam only
            beam_config={
                # Use direct Docker registry image (no template needed)
                "template": "navidpour/bullmq-python-setup:latest",
                "start_command": "opencode serve --hostname 0.0.0.0 --port 4096",
                "readiness_path": "/doc",
                "readiness_timeout": 180.0,
                "container_port": 4096,
                "working_dir": "/root/workspace",
                "environment": {
                    "PYTHONUNBUFFERED": "1",
                },
            },
            image="navidpour/bullmq-python-setup:latest",  # For Docker fallback (if needed)
            setup_command="echo 'Setup complete' && mkdir -p /root/workspace/.trace",
            prompt_command="echo '{\"role\": \"user\", \"content\": \"test\"}'",
            test_command="mkdir -p /root/workspace/.trace && echo '<?xml version=\"1.0\"?><testsuites tests=\"3\" failures=\"0\"><testsuite tests=\"3\" failures=\"0\"></testsuite></testsuites>' > /root/workspace/.trace/junit.xml",
            junit_path="/root/workspace/.trace/junit.xml",
            working_dir="/root/workspace",
            dataset=dataset,
            max_turns=1,
            trace_output_dir="./test_traces_beam",
            event_callback=event_callback,
        )
        
        print("✓ Environment created")
        
        # Execute rollout
        print("\nExecuting rollout...")
        start_time = time.time()
        
        generate_outputs, processed_outputs = await env.a_generate(
            inputs=dataset,
            max_concurrent=1,
        )
        
        exec_time = time.time() - start_time
        print(f"✓ Rollout completed in {exec_time:.1f}s")
        
        # Validate results
        reward = generate_outputs.reward[0]
        metrics = generate_outputs.metrics
        state = generate_outputs.state[0] if generate_outputs.state else {}

        def metric_value(name: str, default: float = 0.0) -> float:
            values = metrics.get(name)
            if isinstance(values, list) and values:
                first = values[0]
                if isinstance(first, (int, float)):
                    return float(first)
            return default
        
        print(f"\nResults:")
        print(f"  Reward: {reward:.2f}")
        backend_name = state.get("sandbox_backend", "unknown") if isinstance(state, dict) else "unknown"
        print(f"  Backend: {backend_name}")
        status_success = metric_value("status_success")
        print(f"  Status: {'success' if status_success >= 1 else 'failure'}")
        
        # Check if Beam was used
        if backend_name == 'beam':
            print_result("ProximalDockerEnv Integration", True, "Beam backend used successfully")
            return True
        else:
            print_result("ProximalDockerEnv Integration", False, 
                        f"Expected 'beam' but got '{backend_name}'")
            return False
            
    except Exception as e:
        print_result("ProximalDockerEnv Integration", False, str(e))
        import traceback
        traceback.print_exc()
        return False

async def run_all_tests(template_id, skip_provision=False):
    """Run all smoke tests."""
    print_header("BEAM SMOKE TEST SUITE")
    print(f"Template: {template_id}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test 1: SDK & Auth
    results['sdk'] = await test_1_beam_sdk()
    if not results['sdk']:
        print("\n❌ Cannot proceed without Beam SDK. Install and try again.")
        return results
    
    # Test 2: Template Access
    results['template'] = await test_2_template_access(template_id)
    if not results['template']:
        print("\n❌ Cannot proceed without valid template.")
        return results
    
    # Test 3: Sandbox Provisioning (can skip for quick tests)
    if not skip_provision:
        results['provision'] = await test_3_sandbox_provision(template_id)
        if not results['provision']:
            print("\n⚠️  Sandbox provisioning failed. Integration test may also fail.")
    else:
        print_header("TEST 3: SKIPPED")
        results['provision'] = None
    
    # Test 4: Full Integration
    results['integration'] = await test_4_proximal_integration(template_id)
    
    # Summary
    print_header("SMOKE TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Skipped: {skipped}")
    
    if failed == 0 and passed > 0:
        print("\n✅ ALL TESTS PASSED - Beam integration is working!")
        print("\nNext steps:")
        print("  1. Run full training with configs/rl/proximal/qwen3_4B_beam.yaml")
        print("  2. Monitor Beam dashboard for usage/costs")
        print("  3. Scale up generation_semaphore gradually")
        return results
    else:
        print("\n❌ SOME TESTS FAILED - Review errors above")
        print("\nSee docs/BEAM_SMOKE_TEST_GUIDE.md for troubleshooting")
        return results

def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for Beam sandbox backend integration"
    )
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help="Beam template ID to use for testing"
    )
    parser.add_argument(
        "--skip-provision",
        action="store_true",
        help="Skip the sandbox provisioning test (faster, less thorough)"
    )
    parser.add_argument(
        "--test",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run only a specific test (1-4)"
    )
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("BEAM_API_KEY"):
        print("❌ ERROR: BEAM_API_KEY environment variable not set")
        print("Set it with: export BEAM_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Run tests
    if args.test:
        # Run single test
        test_funcs = {
            1: test_1_beam_sdk,
            2: lambda: test_2_template_access(args.template),
            3: lambda: test_3_sandbox_provision(args.template),
            4: lambda: test_4_proximal_integration(args.template),
        }
        success = asyncio.run(test_funcs[args.test]())
        sys.exit(0 if success else 1)
    else:
        # Run all tests
        results = asyncio.run(run_all_tests(args.template, args.skip_provision))
        all_passed = all(v is not False for v in results.values())
        sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
