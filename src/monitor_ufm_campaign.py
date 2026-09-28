"""Read-only combined monitor for extraction, full UFM and ablation suite."""
import json
from pathlib import Path
from monitor_ufm_training import main as monitor_main


if __name__=='__main__':
    monitor_main()
    root = Path(__file__).resolve().parents[1]
    marker = root/'runs/ufm_ablation_suite_v1.json'
    print('\nABLATION SUITE')
    print(marker.read_text() if marker.exists() else 'NOT STARTED')
    for output in sorted((root/'runs').glob('ufm_ablation_*_s42_v1')):
        complete = output/'completed.json'
        print(output.name, json.loads(complete.read_text()) if complete.exists() else 'NOT COMPLETE')
