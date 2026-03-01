def test_main_imports():
    try:
        import src.main
        assert True
    except Exception as e:
        assert False, f"Failed to import main: {e}"
