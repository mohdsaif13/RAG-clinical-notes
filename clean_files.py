import hashlib
import json
from pathlib import Path


SOURCE_ROOT_DIRECTORY = Path("./dataset")
OUTPUT_DIRECTORY = Path("./Cleaned_Clinical_Notes")
OUTPUT_PATTERN = "*_cleaned.txt"

SECTION_FIELDS = [
    ("ADMISSION/CHIEF COMPLAINT", "input1"),
    ("PATIENT HISTORY", "input2"),
    ("PAST MEDICAL HISTORY", "input3"),
    ("FAMILY HISTORY", "input4"),
    ("PHYSICAL EXAM", "input5"),
    ("LABS AND IMAGING", "input6"),
]


def has_clinical_value(value):
    if value is None:
        return False

    text = str(value).strip()
    return bool(text) and text.lower() != "none"


def build_output_filename(source_path):
    relative_path = source_path.relative_to(SOURCE_ROOT_DIRECTORY)
    path_hash = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{source_path.stem}_{path_hash}_cleaned.txt"


def clear_previous_outputs():
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)

    removed_count = 0
    for old_file in OUTPUT_DIRECTORY.glob(OUTPUT_PATTERN):
        old_file.unlink()
        removed_count += 1

    return removed_count


def extract_clinical_sections(data):
    cleaned_content = []

    for header, key in SECTION_FIELDS:
        value = data.get(key)
        if has_clinical_value(value):
            cleaned_content.append(f"=== {header} ===")
            cleaned_content.append(str(value).strip())
            cleaned_content.append("")

    return "\n".join(cleaned_content).strip()


def clean_and_save_files():
    if not SOURCE_ROOT_DIRECTORY.exists():
        raise FileNotFoundError(f"Dataset directory not found: {SOURCE_ROOT_DIRECTORY}")

    removed_count = clear_previous_outputs()
    files_processed = 0
    files_skipped = 0

    print(f"Starting search in: {SOURCE_ROOT_DIRECTORY.resolve()}...")
    print(f"Removed {removed_count} previous cleaned file(s).")

    for source_path in SOURCE_ROOT_DIRECTORY.rglob("*.json"):
        try:
            with source_path.open("r", encoding="utf-8") as source_file:
                data = json.load(source_file)

            final_text = extract_clinical_sections(data)

            if not final_text:
                print(f"Skipped empty inputs: {source_path}")
                files_skipped += 1
                continue

            output_path = OUTPUT_DIRECTORY / build_output_filename(source_path)
            output_path.write_text(final_text, encoding="utf-8")

            files_processed += 1
            print(f"Processed: {source_path}")

        except Exception as e:
            print(f"Error reading {source_path}: {e}")
            files_skipped += 1

    print("-" * 30)
    print("Done.")
    print(f"Processed files: {files_processed}")
    print(f"Skipped/Error files: {files_skipped}")
    print(f"Cleaned files are saved in: {OUTPUT_DIRECTORY.resolve()}")


if __name__ == "__main__":
    clean_and_save_files()
