from pathlib import Path
import shutil
import subprocess

import onnx
from onnx import TensorProto,  shape_inference
from ultralytics import YOLO

# ============================================================
# 閰嶇疆
# ============================================================
MODEL_DIR = Path("./models").absolute()
ONNX_DIR = MODEL_DIR / "onnx"
TRT_ONNX_DIR = MODEL_DIR / "trt-onnx"

DEVICE = "cpu"
SIMPLIFY = False
DYNAMIC = False
HALF = False
OPTIMIZE = False
OPSET = 12

MAX_DETS = 16
CONF_THRES = 0.25
IOU_THRES = 0.45


# ============================================================
# 鍩虹宸ュ叿
# ============================================================
def latest_file(directory: Path, pattern: str, exclude_suffixes: tuple[str, ...] = ()) -> Path:
    candidates = [
        p for p in directory.glob(pattern)
        if p.is_file() and not any(p.stem.endswith(suffix) for suffix in exclude_suffixes)
    ]
    if not candidates:
        raise FileNotFoundError(f"鏈壘鍒?{pattern} 鏂囦欢: {directory}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def ask_mode() -> str:
    print("閫夋嫨瀵煎嚭璺緞锛?)
    print("1. 浠庢渶鏂?PT 寮€濮嬪鍑?)
    print("2. 浠庣幇鏈?ONNX 寮€濮嬭浆鎹?)
    choice = input("璇疯緭鍏?1 鎴?2: ").strip()
    if choice not in {"1", "2"}:
        raise ValueError("鏃犳晥閫夋嫨锛屽彧鑳借緭鍏?1 鎴?2銆?)
    return choice


def ask_is_e2e() -> bool:
    choice = input("鏄惁涓?e2e 妯″瀷锛?y/N): ").strip().lower()
    return choice in {"y", "yes"}


# ============================================================
# ONNX 淇ˉ锛氳ˉ output 鐨?shape / dtype
# ============================================================
def _copy_shape(src_tensor_type, dst_tensor_type) -> bool:
    if not src_tensor_type.HasField("shape"):
        return False

    changed = False
    dst_shape = dst_tensor_type.shape
    if len(dst_shape.dim) == 0 and len(src_tensor_type.shape.dim) > 0:
        for dim in src_tensor_type.shape.dim:
            new_dim = dst_shape.dim.add()
            if dim.HasField("dim_value"):
                new_dim.dim_value = dim.dim_value
            elif dim.HasField("dim_param"):
                new_dim.dim_param = dim.dim_param
        changed = True
    return changed


def patch_onnx_metadata(onnx_path: Path) -> Path:
    patched_path = onnx_path.with_name(f"{onnx_path.stem}_patched.onnx")
    if patched_path.exists():
        patched_path.unlink()

    model = onnx.load(str(onnx_path))
    model = shape_inference.infer_shapes(model)

    tensor_info_map = {}
    for value in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if value.type.HasField("tensor_type"):
            tensor_info_map[value.name] = value.type.tensor_type

    output_dtype_map = {
        "num_dets": TensorProto.INT32,
        "det_boxes": TensorProto.FLOAT,
        "det_scores": TensorProto.FLOAT,
        "det_classes": TensorProto.INT32,
    }

    changed = False
    new_outputs = []

    for output in model.graph.output:
        tensor_type = output.type.tensor_type
        output_changed = False

        ref_tensor_type = tensor_info_map.get(output.name)

        if tensor_type.elem_type == 0:
            if output.name in output_dtype_map:
                tensor_type.elem_type = output_dtype_map[output.name]
                output_changed = True
            elif ref_tensor_type is not None and ref_tensor_type.elem_type != 0:
                tensor_type.elem_type = ref_tensor_type.elem_type
                output_changed = True

        if ref_tensor_type is not None:
            if _copy_shape(ref_tensor_type, tensor_type):
                output_changed = True

        if output_changed:
            changed = True

        new_outputs.append(output)

    if not changed:
        return onnx_path

    del model.graph.output[:]
    model.graph.output.extend(new_outputs)
    onnx.save(model, str(patched_path))
    print(f"鉁?宸蹭慨琛?ONNX 杈撳嚭鍏冧俊鎭? {patched_path}")
    return patched_path


# ============================================================
# PT -> 鏅€?ONNX
# ============================================================
def export_plain_onnx(pt_path: Path) -> Path:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/2] 姝ｅ湪瀵煎嚭鏅€?ONNX...\n妯″瀷: {pt_path}")
    model = YOLO(str(pt_path))

    model.export(
        format="onnx",
        device=DEVICE,
        simplify=SIMPLIFY,
        dynamic=DYNAMIC,
        half=HALF,
        optimize=OPTIMIZE,
        opset=OPSET,
    )

    default_output = pt_path.with_suffix(".onnx")
    if not default_output.exists():
        raise FileNotFoundError(f"鏈壘鍒?Ultralytics 瀵煎嚭鐨?ONNX: {default_output}")

    output_path = ONNX_DIR / f"{pt_path.stem}.onnx"
    if output_path.exists():
        output_path.unlink()

    shutil.move(str(default_output), str(output_path))
    print(f"鉁?鏅€?ONNX 瀵煎嚭瀹屾垚: {output_path}")
    return output_path


# ============================================================
# 鏅€?ONNX -> TRT-YOLO ONNX
# ============================================================
def export_trtyolo_onnx(onnx_path: Path, is_e2e: bool) -> Path:
    TRT_ONNX_DIR.mkdir(parents=True, exist_ok=True)

    fixed_onnx_path = patch_onnx_metadata(onnx_path)
    output_path = TRT_ONNX_DIR / f"{onnx_path.stem}_trtyolo.onnx"
    if output_path.exists():
        output_path.unlink()

    print(f"\n姝ｅ湪杞崲 TRT-YOLO ONNX...\n妯″瀷: {fixed_onnx_path}")
    export_cmd = [
        "trtyolo-export",
        "-i",
        str(fixed_onnx_path),
        "-o",
        str(output_path),
        "--max-dets",
        str(MAX_DETS),
        "--conf-thres",
        str(CONF_THRES),
    ]

    if not is_e2e:
        export_cmd.extend(
            [
                "--iou-thres",
                str(IOU_THRES),
            ]
        )

    try:
        subprocess.run(export_cmd, check=True)
    except FileNotFoundError:
        raise FileNotFoundError("鎵句笉鍒?trtyolo-export锛岃纭瀹冨凡瀹夎骞跺湪 PATH 涓€?)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"trtyolo-export 杞崲澶辫触锛岄€€鍑虹爜: {exc.returncode}") from exc

    print(f"鉁?TRT-YOLO ONNX 杞崲瀹屾垚: {output_path}")
    return output_path


# ============================================================
# 涓绘祦绋?
# ============================================================
def main() -> None:
    mode = ask_mode()
    is_e2e = ask_is_e2e()

    if mode == "1":
        pt_path = latest_file(MODEL_DIR, "*.pt")
        plain_onnx = export_plain_onnx(pt_path)
        trt_onnx = export_trtyolo_onnx(plain_onnx, is_e2e)

        print("\n鍏ㄩ儴瀹屾垚锛?)
        print(f"鏅€?ONNX: {plain_onnx}")
        print(f"TRT-YOLO ONNX: {trt_onnx}")
        return

    onnx_path = latest_file(
        ONNX_DIR,
        "*.onnx",
        exclude_suffixes=("_patched", "_trtyolo"),
    )
    trt_onnx = export_trtyolo_onnx(onnx_path, is_e2e)

    print("\n鍏ㄩ儴瀹屾垚锛?)
    print(f"TRT-YOLO ONNX: {trt_onnx}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n瀵煎嚭澶辫触: {exc}")

