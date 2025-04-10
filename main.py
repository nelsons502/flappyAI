import argparse
import subprocess
import socket
import time

def start_game_server(mode):
    print("Starting game server...")
    return subprocess.Popen(["python", "flappy_bird.py", "--mode", mode])

def start_ai_server(mode, model_path=None, data_path=None):
    print(f"[INFO] Starting AI server in '{mode}' mode...")
    ai_args = ["python", "smart_flap.py", "--mode", mode]
    if model_path:
        ai_args += ["--model_path", model_path]
    if data_path:
        ai_args += ["--data_path", data_path]
    return subprocess.Popen(ai_args)

def main():
    parser = argparse.ArgumentParser(description="Control Flappy Bird Training and Gameplay.")
    parser.add_argument("mode", choices=["train", "train_data", "continue", "play"], help="Mode to run the AI.")
    parser.add_argument("--model_path", type=str, help="Path to the model file.")
    parser.add_argument("--data_path", type=str, help="Path to the pre-collected data.")
    args = parser.parse_args()

    ai_process = start_ai_server(args.mode, args.model_path, args.data_path)
    time.sleep(2)  # Wait a moment for the AI server to start
    game_process = start_game_server(args.mode)

    try:
        print("Both processes are started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping processes...")
        game_process.terminate()
        ai_process.terminate()

if __name__ == "__main__":
    main()