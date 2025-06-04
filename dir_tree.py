import os

# Lista di cartelle/file da escludere
exclude_dirs = [
    ".git",
    ".idea",
    "__pycache__",
    "dir_tree.py",
    "dist",
]  # Puoi aggiungere altri nomi


def print_tree(path, prefix=""):
    """
    Stampa l'albero della directory, mostrando solo file .py e .sh.
    Esclude le cartelle definite in `exclude_dirs`.
    Mostra prima i file, poi le directory.

    :param path: Directory da esplorare.
    :param prefix: Prefisso per indentazione ricorsiva.
    """
    if not os.path.exists(path):
        print(f"La directory {path} non esiste.")
        return

    items = [item for item in sorted(os.listdir(path)) if item not in exclude_dirs]

    # Separiamo file e directory
    files = [
        item
        for item in items
        if os.path.isfile(os.path.join(path, item)) and item.endswith((".py", ".sh"))
    ]
    dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]

    combined = files + dirs  # File prima, poi cartelle

    for idx, item in enumerate(combined):
        full_path = os.path.join(path, item)
        connector = "└── " if idx == len(combined) - 1 else "├── "

        if os.path.isfile(full_path):
            print(f"{prefix}{connector}{item}")
        elif os.path.isdir(full_path):
            print(f"{prefix}{connector}{item}/")
            new_prefix = prefix + ("    " if idx == len(combined) - 1 else "│   ")
            print_tree(full_path, new_prefix)


# Percorso root del progetto
directory_path = (
    "/home/cristiano.massaroni/work_projects/synt_data/synthetic_med_models"
)
print(f"\n📂 {directory_path}")
print_tree(directory_path)
