import os


def list_files_in_folder(folder_path):
    """List all files in folder"""
    file_names = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_names.append(file)
    return file_names

def sanitize_entities(entities):
    """Formating the data."""
    sanitized = []
    for ent in entities:
        sanitized.append({
            "word": ent["word"],
            "entity_group": ent["entity_group"],
            "score": float(ent["score"]),  # Convert to native float
            "start": ent["start"],
            "end": ent["end"]
        })
    return sanitized




