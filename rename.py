import os


def rename_files(root: str, prefix: str):
    for i, f in enumerate(os.listdir(root)):
        try:
            if f.endswith(".m4a"):
                os.rename(os.path.join(root, f), os.path.join(root, f"{prefix}{i}.m4a"))
            if f.endswith(".wav"):
                os.rename(os.path.join(root, f), os.path.join(root, f"{prefix}{i}.wav"))
        except Exception as e:
            print(e)


if __name__ == '__main__':
    root_dir = "data/Wave"
    for i, d in enumerate(os.listdir(root_dir)):
        rename_files(os.path.join(root_dir, d), f"")
