import kagglehub
from pathlib import Path

def download_kaggle_dataset(dataset_name, target_path):
    downloaded_path = Path(kagglehub.dataset_download(dataset_name))
    print('Downloaded dataset to:', downloaded_path)

    if target_path.is_symlink():
        print('Symbolic link already exists: ', target_path)
        return

    target_path.symlink_to(downloaded_path)
    print(f"Created symbolic link: {target_path} -> {downloaded_path}")


def main():
    data_dir = Path('data/')
    data_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ('raddar/chest-xrays-indiana-university', 'indiana'),
        ('tawsifurrahman/covid19-radiography-database', 'covid'),
        ('tawsifurrahman/tuberculosis-tb-chest-xray-dataset', 'tuberculosis'),
        ('nih-chest-xrays/data', 'nih'),
    ]

    for dataset_name, target_name in datasets:
        download_kaggle_dataset(dataset_name, data_dir / target_name)


if __name__ == "__main__":
    main()
