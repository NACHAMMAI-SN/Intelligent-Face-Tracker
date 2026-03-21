"""Application entrypoint for the intelligent face tracker."""

from __future__ import annotations

from src.pipeline import Pipeline


def _print_stats(stats: dict[str, object]) -> None:
    """Print pipeline stats in a readable format."""
    print("\nPipeline finished. Final stats:")
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        print(f"- {label}: {value}")


def main() -> None:
    """Build and run the face-tracking pipeline."""
    pipeline = None
    try:
        pipeline = Pipeline.from_config("config.json")
        stats = pipeline.run()
        _print_stats(stats)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user (Ctrl+C).")
        # If interrupted before/around run completion, still print known stats.
        if pipeline is not None:
            _print_stats(pipeline.stats.to_dict())
    except Exception as exc:
        print(f"\nFatal error while running pipeline: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
