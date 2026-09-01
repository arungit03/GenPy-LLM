import subprocess
import tempfile
import os
import time
import textwrap

EXECUTION_TIMEOUT = 5  # seconds

def run_code_in_subprocess(code: str, test_code: str, timeout: int = EXECUTION_TIMEOUT) -> tuple[bool, str]:
    """Safely execute generated code + test assertions in a subprocess with timeout.
    
    Args:
        code: Generated Python code to test.
        test_code: Test assertions (e.g., "assert add(1,2)==3").
        timeout: Maximum execution time in seconds.
        
    Returns:
        (passed, output_or_error) tuple.
    """
    full_code = textwrap.dedent(code) + "\n\n" + textwrap.dedent(test_code)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(full_code)
        tmpfile = f.name
        
    # Strip environment variables for slight security improvement
    # NOTE: This does not prevent filesystem/network access.
    # True isolation requires Docker or a VM.
    safe_env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", "") # Required on Windows
    }
        
    try:
        result = subprocess.run(
            [sys.executable, tmpfile],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=safe_env
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {timeout}s"
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(tmpfile):
            os.unlink(tmpfile)


def functional_pass_rate(test_cases: list[dict]) -> dict:
    """Run all test cases and return pass rates.
    
    Args:
        test_cases: List of dicts with keys: 'code', 'tests', 'prompt'
        
    Returns:
        dict with 'passed', 'failed', 'total', 'pass_rate' keys.
    """
    passed = 0
    failed = 0
    errors = []
    
    for case in test_cases:
        success, msg = run_code_in_subprocess(case['code'], case['tests'])
        if success:
            passed += 1
        else:
            failed += 1
            errors.append({'prompt': case.get('prompt', ''), 'error': msg})
            
    total = passed + failed
    return {
        'passed': passed,
        'failed': failed,
        'total': total,
        'pass_rate': passed / total if total else 0.0,
        'errors': errors,
    }
