"""Tests for Serve CLI."""

from SneppX_ALG.interface_bindings.serve_cli import main as serve_main


def test_serve_cli_import():
    from SneppX_ALG.interface_bindings import serve_cli
    assert serve_cli is not None


def test_serve_cli_has_main():
    assert callable(serve_main)


if __name__ == "__main__":
    import sys
    locals_ = {k: v for k, v in locals().items() if k.startswith("test_")}
    passed = 0
    failed = 0
    for name, fn in sorted(locals_.items()):
        try:
            fn()
            print(f"  PASS {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {name}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    sys.exit(failed)
