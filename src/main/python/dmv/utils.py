from pathlib import Path

def request_run_identifier(log_dir: Path):
    while True:
        print('Enter a run identifier: ')
        identifier = input()

        identifier.replace(' ', '-')

        if not (log_dir / identifier).exists():
            return identifier
        else:
            print(f'Path \'{log_dir / identifier}\' exists already')