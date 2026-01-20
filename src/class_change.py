import argparse
import re
import sys
from pathlib import Path

# Script to fix indexes after merge of datasets


def patch_labels(root: Path):
    counter = 0
    for txt in root.rglob("*.txt"):
        try:
            with txt.open("r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"SKIP  {txt}  (read error: {e})", file=sys.stderr)
            continue

        new_lines = []
        changed = False
        for line in lines:
            # YOLO line: <class_id> <x_center> <y_center> <width> <height>
            parts = line.strip().split()
            if not parts:  # empty line
                new_lines.append(line)
                continue
            # Reorder class indexes
            if parts[0] == "10":
                parts[0] = "0"
                changed = True
            elif parts[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                parts[0] = str(int(parts[0]) + 1)
            new_lines.append(" ".join(parts) + "\n")

        if changed:
            try:
                with txt.open("w", encoding="utf-8") as f:
                    f.writelines(new_lines)
                counter += 1
                print(f"FIXED {txt}")
            except Exception as e:
                print(f"ERROR {txt}  (write error: {e})", file=sys.stderr)

    print(f"\nAll done – patched {counter} files.")


if __name__ == "__main__":
    patch_labels(Path("datasets/seven_seg_merge"))
