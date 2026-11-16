"""Write every IPU kernel of ML-KEM-768 as the exact assembly the emulator runs, plus the launch sequences.

    python ipu/dump_asm.py [out_dir]          (default: asm/ next to ipu/)

out_dir/<kernel>.asm   the program text, exactly as passed to the assembler (one line per instruction word;
                       ' ; ' separates the slots of a word, ';;' ends it)
out_dir/launches.json  for each operation (keygen, encaps, decaps) the kernels in launch order, and for each
                       kernel its category, instruction words and the CR values the host writes before launching
                       it (LRs are reset to 0). A kernel's constant XMEM rows are written once before the first
                       launch and are not part of this listing.
Each file is checked: assembling it again gives the same machine words the KEM runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "kernels")]
import emu  # noqa: E402
import mlkem768_ipu as M  # noqa: E402

OPS = ("keygen", "encaps", "decaps")


def cr_value(v):
    return f"0x{v[1] & 0xFFFFFFFF:08x}" if isinstance(v, tuple) else float(v)


def main(out: Path) -> None:
    kem = M.IpuMlKem768(emu.StrictSession("nx8"))
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.asm"):
        f.unlink()
    kernels, order = {}, {}
    for op in OPS:
        seq = getattr(kem, f"seq_{op}")
        order[op] = [k["name"] for k in seq]
        for k in seq:
            prev = kernels.get(k["name"])
            if prev is not None:
                assert prev is k or prev["asm"] == k["asm"], f"two different programs named {k['name']}"
                continue
            kernels[k["name"]] = k
    for name, k in sorted(kernels.items()):
        text = k["asm"] if k["asm"].endswith("\n") else k["asm"] + "\n"
        (out / f"{name}.asm").write_text(text)
        assert emu.assemble((out / f"{name}.asm").read_text()) == list(k["words"]), name
    meta = {name: dict(category=k["cat"], words=len(k["words"]),
                       crs={cr: cr_value(v) for cr, v in sorted(k.get("crs", {}).items(), key=lambda x: int(x[0][2:]))})
            for name, k in sorted(kernels.items())}
    (out / "launches.json").write_text(json.dumps(dict(launch_order=order, kernels=meta), indent=1) + "\n")
    print(f"{len(kernels)} kernels, {sum(len(v) for v in order.values())} launches "
          f"({', '.join(f'{op} {len(order[op])}' for op in OPS)}) -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "asm")
