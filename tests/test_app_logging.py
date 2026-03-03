from src.app_logging import configure_logging


def test_configure_logging_writes_to_file(tmp_path):
    log_file = tmp_path / "logs" / "app.log"
    logger = configure_logging(level="INFO", log_file=str(log_file))
    logger.info("hello-log")

    for handler in logger.handlers:
        handler.flush()

    content = log_file.read_text(encoding="utf-8")
    assert "hello-log" in content
