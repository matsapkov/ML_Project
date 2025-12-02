from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import datetime

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs" / "sb3"

scripts = [
    * (BASE_DIR / "frozen-lake").glob("*.py"),
    * (BASE_DIR / "taxi").glob("*.py"),
    * (BASE_DIR / "lunar-lander").glob("*.py"),
]

def run_script(script_path: Path):
    print(f"[RUN] {script_path.name}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    print(f"[DONE] {script_path.name}")

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return script_path.name

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    print(f"[{start_time}] Запуск всех процессов...")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Найдено {len(scripts)} скриптов:")
    for s in scripts:
        print("  -", s.name)

    print("\nЗапуск процессов...\n")

    results = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(run_script, s): s for s in scripts}

        for future in as_completed(futures):
            script_name = futures[future].name
            try:
                future.result()
                results.append((script_name, "OK"))
            except Exception as e:
                results.append((script_name, f"ERROR: {e}"))

    end_time = datetime.datetime.now()
    print(f"\n[{end_time}] Все процессы завершены.")

    print("\n==================== ИТОГ ====================")
    for name, status in results:
        print(f"{name:25} — {status}")
    print("==============================================\n")

    print("TensorBoard:")
    print(f"  tensorboard --logdir \"{LOG_DIR}\" --port 6006")
    print("Откройте: http://localhost:6006\n")
