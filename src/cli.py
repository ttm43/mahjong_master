import argparse
import json

from src.main import main as run_app
from src.ml.placeholders import (
    check_data_scaffold,
    check_training_scaffold,
    init_data_scaffold,
    init_training_scaffold,
    run_data_collection,
    run_model_training,
)
from src.preflight import run_preflight_checks, run_preflight_report


def _print_missing(prefix, missing):
    if not missing:
        print(f"{prefix} scaffold is ready")
        return
    print(f"{prefix} scaffold is missing {len(missing)} item(s):")
    for item in missing:
        print(f"- {item}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="mahjong-master",
        description="Mahjong Master CLI",
        epilog=(
            "Exit codes:\n"
            "  0 = success\n"
            "  1 = usage/help shown\n"
            "  2 = invalid flag combination or placeholder command\n"
            "  3 = check/preflight failed"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run realtime overlay pipeline")
    run_parser.add_argument("--config", help="Path to app config file")
    preflight_parser = subparsers.add_parser("preflight", help="Validate runtime prerequisites")
    preflight_parser.add_argument("--config", help="Path to app config file")
    preflight_parser.add_argument("--json", action="store_true", help="Output preflight report as JSON")

    collect_parser = subparsers.add_parser("collect-data", help="Data collection placeholder")
    collect_parser.add_argument("--workspace", default=".", help="Workspace root for generated files")
    collect_parser.add_argument("--init-scaffold", action="store_true", help="Create data scaffold layout")
    collect_parser.add_argument("--check", action="store_true", help="Check data scaffold completeness")
    collect_parser.add_argument("--dry-run", action="store_true", help="Preview scaffold files without creating")

    train_parser = subparsers.add_parser("train-models", help="Model training placeholder")
    train_parser.add_argument("--workspace", default=".", help="Workspace root for generated files")
    train_parser.add_argument("--init-scaffold", action="store_true", help="Create training scaffold layout")
    train_parser.add_argument("--check", action="store_true", help="Check training scaffold completeness")
    train_parser.add_argument("--dry-run", action="store_true", help="Preview scaffold files without creating")

    args = parser.parse_args(argv)

    if args.command == "run":
        run_app(config_path=args.config)
        return 0

    if args.command == "preflight":
        if args.json:
            report = run_preflight_report(config_path=args.config)
            print(json.dumps(report, ensure_ascii=False))
            return 3 if report["issues"] else 0

        issues, warnings = run_preflight_checks(config_path=args.config)
        if warnings:
            print("Preflight completed with warnings:")
            for warning in warnings:
                print(f"- {warning}")
        if issues:
            print("Preflight failed:")
            for issue in issues:
                print(f"- {issue}")
            return 3
        if not warnings:
            print("Preflight completed successfully")
        return 0

    if args.command == "collect-data":
        if args.check and args.init_scaffold:
            print("--check and --init-scaffold cannot be used together")
            return 2

        if args.check:
            missing = check_data_scaffold(args.workspace)
            _print_missing("Data", missing)
            return 0 if not missing else 3

        if args.init_scaffold:
            changed = init_data_scaffold(args.workspace, dry_run=args.dry_run)
            mode = "previewed" if args.dry_run else "initialized"
            print(f"Data scaffold {mode} at {args.workspace} ({len(changed)} item(s))")
            return 0

        try:
            run_data_collection()
        except NotImplementedError as exc:
            print(exc)
            return 2

    if args.command == "train-models":
        if args.check and args.init_scaffold:
            print("--check and --init-scaffold cannot be used together")
            return 2

        if args.check:
            missing = check_training_scaffold(args.workspace)
            _print_missing("Training", missing)
            return 0 if not missing else 3

        if args.init_scaffold:
            changed = init_training_scaffold(args.workspace, dry_run=args.dry_run)
            mode = "previewed" if args.dry_run else "initialized"
            print(f"Training scaffold {mode} at {args.workspace} ({len(changed)} item(s))")
            return 0

        try:
            run_model_training()
        except NotImplementedError as exc:
            print(exc)
            return 2

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
