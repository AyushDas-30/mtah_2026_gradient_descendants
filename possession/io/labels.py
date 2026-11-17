"""Read and write frame,label CSVs, with the same checks the official validator applies."""

from pathlib import Path

import numpy as np

from ..schema import CHAR_TO_CODE, LABEL_CHARS


def read_labels(path):
    """Codes array from a frame,label CSV. Raises with a one-line reason on a malformed file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    header = [c.strip().lower() for c in lines[0].split(",")]
    if header != ["frame", "label"]:
        raise ValueError("{}: header {} != ['frame', 'label']".format(path, header))
    codes = np.empty(len(lines) - 1, dtype=np.int8)
    for i, line in enumerate(lines[1:]):
        frame, label = line.split(",")
        if int(frame) != i:
            raise ValueError("{}: row {} has frame {}".format(path, i, frame))
        label = label.strip().upper()
        if label not in CHAR_TO_CODE:
            raise ValueError("{}: row {} label {!r}".format(path, i, label))
        codes[i] = CHAR_TO_CODE[label]
    return codes


def write_labels(path, codes, n_expected=None):
    codes = np.asarray(codes)
    if n_expected is not None and len(codes) != n_expected:
        raise ValueError("refusing to write {} rows, expected {}".format(len(codes), n_expected))
    if codes.min() < 0 or codes.max() > 3:
        raise ValueError("label codes outside 0..3")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chars = LABEL_CHARS[codes.astype(np.int64)]
    with open(path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, c) for i, c in enumerate(chars))
    return path
