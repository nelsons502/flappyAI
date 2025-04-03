import argparse
import subprocess
import socket
import time

def start_game_server():
    print("Starting game server...")
    return subprocess.Popen(["python", "flappy_bird.py"])

def start_ai_server(mode, model_path=None, data_path=None):
    print(f"Starting AI server in {mode} mode...")
    args = ["jupyter", "nbconvert", "--to", "script", "smart_flap.ipynb", "--output", "smart_flap.py"]
    subprocess.run(args)
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

    game_procss = start_game_server()

    time.sleep(3)
    ai_process = start_ai_server(args.mode, args.model_path, args.data_path)

    try:
        print("Both processes are started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping processes...")
        game_procss.terminate()
        ai_process.terminate()

if __name__ == "__main__":
    main()