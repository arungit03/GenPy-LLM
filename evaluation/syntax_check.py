def check_syntax(code: str) -> tuple[bool, str]:
    """Check if a Python code string is syntactically valid.
    
    Args:
        code: Python source code string.
        
    Returns:
        (is_valid, error_message) tuple.
    """
    try:
        compile(code, "<generated>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def check_syntax_batch(codes: list[str]) -> list[tuple[bool, str]]:
    """Check syntax for a batch of code strings."""
    return [check_syntax(code) for code in codes]

def syntax_pass_rate(codes: list[str]) -> float:
    """Return the fraction of code strings that are syntactically valid."""
    if not codes:
        return 0.0
    results = check_syntax_batch(codes)
    passed = sum(1 for valid, _ in results if valid)
    return passed / len(codes)
