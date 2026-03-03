def test_main_imports():
    try:
        module = __import__("src.main", fromlist=["main"])
        assert module is not None
    except Exception as e:
        assert False, f"Failed to import main: {e}"
