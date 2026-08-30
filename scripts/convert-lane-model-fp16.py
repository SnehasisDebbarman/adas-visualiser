import sys
from pathlib import Path


def main():
    if len(sys.argv) != 3:
        print("usage: convert-lane-model-fp16.py INPUT OUTPUT")
        return 2

    source = Path(sys.argv[1])
    target = Path(sys.argv[2])

    try:
        import onnx
        from onnxconverter_common import float16
    except Exception as exc:
        print(f"FP16 conversion dependencies unavailable: {exc}")
        return 3

    print(f"Converting {source.name} to FP16…")
    model = onnx.load(str(source))
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    onnx.save(model_fp16, str(target))

    if not target.exists() or target.stat().st_size < 1_000_000:
        raise RuntimeError("FP16 model output is unexpectedly small")

    print(
        f"FP16 UFLDv2 ready: {target.name} "
        f"({target.stat().st_size / 1024 / 1024:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
