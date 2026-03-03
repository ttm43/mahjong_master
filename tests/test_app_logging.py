from src.app_logging import configure_logging


def test_configure_logging_writes_to_file(tmp_path):
    log_file = tmp_path / "logs" / "app.log"
    logger = configure_logging(level="INFO", log_file=str(log_file))
    logger.info("hello-log")

    for handler in logger.handlers:
        handler.flush()

    content = log_file.read_text(encoding="utf-8")
    assert "hello-log" in content


def test_configure_logging_writes_json_lines(tmp_path):
    log_file = tmp_path / "logs" / "app.jsonl"
    logger = configure_logging(level="INFO", log_file=str(log_file), log_format="json")
    logger.info("runtime-start", extra={"event": "runtime_start", "phase": "init"})

    for handler in logger.handlers:
        handler.flush()

    first_line = log_file.read_text(encoding="utf-8").splitlines()[0]
    payload = __import__("json").loads(first_line)

    assert payload["message"] == "runtime-start"
    assert payload["level"] == "INFO"
    assert payload["event"] == "runtime_start"
    assert payload["phase"] == "init"


def test_configure_logging_rotates_file(tmp_path):
    log_file = tmp_path / "logs" / "rotate.log"
    logger = configure_logging(
        level="INFO",
        log_file=str(log_file),
        rotate_bytes=220,
        backup_count=1,
    )

    for idx in range(80):
        logger.info("line-%03d-%s", idx, "x" * 16)

    for handler in logger.handlers:
        handler.flush()

    rotated = log_file.with_name(log_file.name + ".1")
    assert log_file.exists()
    assert rotated.exists()
