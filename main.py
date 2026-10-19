"""Start or stop the LLM checker (web dashboard and API) with the 4d model.

    python main.py [start] [--host H] [--port P] [--reload] [--no-browser] [--background]
    python main.py stop

The model weights live outside the repo: set CORPUS_DIR to the folder that holds them (see README, Quick start).
"""
import argparse
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PID_FILE = ROOT / ".llm-checker.pid"
MODEL_4D = "deberta-v3-large-lora-authorship-4d"
CHECK = f"""
import sys
from src.authorship import registry
record = registry.load()
if not record or record.get("model_id") != "{MODEL_4D}":
    sys.exit("The registry doesn't hold the 4d model.")
ok, reason = registry.verify_files(record)
if not ok:
    sys.exit(f"The 4d model can't be served: {{reason}}. Looked in {{registry.artifact_dir(record)}}. "
             "Set CORPUS_DIR to the folder that holds the model.")
"""


def stop() -> None:
    if not PID_FILE.exists():
        sys.exit("No background checker is recorded. A checker started in a terminal stops with Ctrl+C.")
    pid = int(PID_FILE.read_text())
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
    else:
        import signal
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    PID_FILE.unlink()
    print(f"Stopped the checker (pid {pid}).")


def start(args) -> None:
    env = dict(os.environ)
    if subprocess.run([sys.executable, "-c", CHECK], cwd=ROOT, env=env).returncode:
        sys.exit(1)
    print(f"Serving {MODEL_4D} from {ROOT}")
    command = [sys.executable, "-m", "uvicorn", "src.api:app", "--host", args.host, "--port", str(args.port)]
    if args.reload:
        command.append("--reload")
    url = f"http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}"

    if args.background:
        if PID_FILE.exists():
            sys.exit(f"Already recorded as running ({PID_FILE.name}). Run `python main.py stop` first.")
        flags = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0
        log = open(ROOT / "llm-checker.log", "ab")
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log, creationflags=flags)
        PID_FILE.write_text(str(proc.pid))
        print(f"Running in the background at {url} (pid {proc.pid}, log llm-checker.log). Stop it with: python main.py stop")
        return

    if not args.no_browser:
        threading.Timer(3.0, webbrowser.open, args=(url,)).start()
    try:
        sys.exit(subprocess.run(command, cwd=ROOT, env=env).returncode)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM checker with the 4d model")
    parser.add_argument("action", nargs="?", default="start", choices=["start", "stop"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    parser.add_argument("--no-browser", action="store_true", help="don't open the dashboard")
    parser.add_argument("--background", action="store_true", help="run without a terminal; stop with `stop`")
    args = parser.parse_args()
    stop() if args.action == "stop" else start(args)


if __name__ == "__main__":
    main()
