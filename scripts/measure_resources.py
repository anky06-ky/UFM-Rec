import argparse
import subprocess
import sys
import time
import psutil
try:
    import pynvml
    has_nvml = True
except ImportError:
    has_nvml = False

def main():
    p = argparse.ArgumentParser(description="Measure peak RAM and VRAM during a subprocess.")
    p.add_argument("command", nargs=argparse.REMAINDER, help="Command to run")
    p.add_argument("--interval", type=float, default=0.5, help="Polling interval in seconds")
    args = p.parse_args()

    if not args.command:
        p.error("Provide a command to run.")

    if has_nvml:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
    else:
        device_count = 0

    peak_ram = 0
    peak_vram = [0] * device_count

    print(f"Starting process: {' '.join(args.command)}")
    start_time = time.time()
    process = subprocess.Popen(args.command)

    try:
        ps_proc = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                # Include children (e.g. dataloaders) in RAM measurement
                mem_info = ps_proc.memory_info().rss
                for child in ps_proc.children(recursive=True):
                    try:
                        mem_info += child.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak_ram = max(peak_ram, mem_info)
            except psutil.NoSuchProcess:
                break
            
            if has_nvml:
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    peak_vram[i] = max(peak_vram[i], mem_info.used)
            
            time.sleep(args.interval)
    except KeyboardInterrupt:
        process.terminate()
        process.wait()

    end_time = time.time()
    if has_nvml:
        pynvml.nvmlShutdown()

    print(f"\n--- Resource Usage Report ---")
    print(f"Duration: {end_time - start_time:.2f} seconds")
    print(f"Exit Code: {process.returncode}")
    print(f"Peak RAM: {peak_ram / (1024**3):.2f} GiB")
    for i, vram in enumerate(peak_vram):
        print(f"Peak VRAM (GPU {i}): {vram / (1024**3):.2f} GiB")

if __name__ == "__main__":
    main()
