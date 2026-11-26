import os
import time

import psutil


def main():

    process = psutil.Process(os.getpid())

    memory_before = (
        process.memory_info().rss / (1024 ** 2)
    )

    start_time = time.perf_counter()

    from poc.target.application import process_order
    from poc.target.metrics import ExtractionFailure

    print("=== Target Application PoC ===")

    try:
        result = process_order()
    except ExtractionFailure as exc:
        if exc.__cause__ is not None:
            raise exc.__cause__ from None
        raise

    print("[PASS] Document loaded")

    print("[PASS] AI extraction completed")
    print("[PASS] Schema validation completed")

    print("[PASS] Report saved")

    elapsed = time.perf_counter() - start_time

    memory_after = (
        process.memory_info().rss / (1024 ** 2)
    )

    print("\nExtracted data:")
    print(result.order.model_dump_json(indent=2))

    print("\nPerformance metrics:")
    print(f"Execution time: {elapsed:.2f} seconds")
    print(f"Memory before: {memory_before:.2f} MiB")
    print(f"Memory after: {memory_after:.2f} MiB")

    print(
        f"Memory difference: "
        f"{memory_after - memory_before:.2f} MiB"
    )

    print(f"\nReport location: {result.report_path}")


if __name__ == "__main__":
    main()
