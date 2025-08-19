import json
import argparse
import subprocess
import os
from mxmoe.kernels.gen_workload import MODEL_ID_TO_LAYERS
from mxmoe.kernels.tile_config import get_gpu_info
from mxmoe.kernels.compose_kernel import TemplateGenerator, KernelType
from project_config import *


def get_qcfg_list(qcfg_file: str, target_layer: int) -> str:
    with open(qcfg_file, "r") as f:
        qcfg = json.load(f)

    LT = qcfg.pop("LT", None)

    qcfg_set = set()
    for layer_idx, v in qcfg.items():
        layer_idx = int(layer_idx)
        if target_layer != -1 and layer_idx != target_layer: continue

        for exp_idx, qexp_cfg in v["experts"].items():
            for _, qlinear_cfg in qexp_cfg.items():
                wbits = qlinear_cfg["w_bits"]
                abits = qlinear_cfg["a_bits"]
                gsize = qlinear_cfg["w_gsize"]
                sym   = "sym" if qlinear_cfg["w_sym"] else "asym"
                qcfg_set.add(f"w{wbits}a{abits}_g{gsize}_{sym}")
    return qcfg_set


if __name__ == "__main__":
    model_id = "qwen2_moe"
    parser = argparse.ArgumentParser(description="Bench workloads for groupgemm kernel (No Python Interface).")
    parser.add_argument("--model", type=str, default=model_id, help="Model ID.")
    parser.add_argument("--dataset", type=str, default="wiki2", help="Dataset ID.")
    parser.add_argument("--qconfig", type=str, default=None, help="Path to the quantization config file.")
    parser.add_argument("--tile_config", type=str, default=None, help="Path to the tile config file.")
    parser.add_argument("--qstr", type=str, default=None, help="Short string to represent quantization config.")
    parser.add_argument("--bs", type=int, default=512, help="Batch size.")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index.")

    args = parser.parse_args()
    model_id = args.model
    dataset  = args.dataset
    bs       = args.bs
    layer = args.layer

    if args.qconfig is not None:
        qconfig  = f" --qconfig {args.qconfig}"
        qcfg_list= get_qcfg_list(args.qconfig, args.layer)
        workload_suffix = "-"+args.qconfig.split("wbits")[1].split(".json")[0]+".json" if args.qconfig != "" else ".json"
    elif args.qstr is not None:
        qconfig  = f" --qstr {args.qstr}"
        qcfg_list = [args.qstr]
        workload_suffix = f"-{args.qstr}.json"
    else:
        qconfig  = ""
        qcfg_list = ["fp16"]
        workload_suffix = f"-fp16.json"

    workload_path = f"{CUR_DIR}/out/workloads/{model_id}-{dataset}-{bs}{workload_suffix}"
    
    if layer == -1:
        layers = MODEL_ID_TO_LAYERS[model_id]
    else:
        layers = [layer]

    for layer in layers:
        print(f"🚀 Processing Layer {layer}...")
        
        # 1. Generate workloads
        print("📊 Step 1: Generating workloads...")
        handle = subprocess.Popen(
            (
            f"python mxmoe/kernels/gen_workload.py"
            f" --model {model_id}"
            f" --dataset {dataset}"
            f" --length {bs}"
            f" --save {workload_path}"
            f" --layer {layer}"
            f"{qconfig}"
            ),
            shell=True,
            env=os.environ.copy(),
        )
        handle.wait()

        # 2. Generate and compile kernel (NO PYTHON INTERFACE!)
        print("🔧 Step 2: Generating CUDA kernels...")
        print(f"qcfg_list: {qcfg_list}")
        gen_dir = f"{CUR_DIR}/mxmoe/kernels/src/generated/"
        
        # 清理之前生成的文件
        if os.path.exists(gen_dir):
            file_lists = [f"{gen_dir}/{x}" for x in os.listdir(gen_dir)]
            for file in file_lists:
                os.remove(file)
        
        gpu_info = get_gpu_info()
        
        tile_configs = None
        if args.tile_config is not None:
            with open(args.tile_config, "r") as f:
                tile_configs = json.load(f)
            
        generator = TemplateGenerator(gpu_info["cc"], qcfg_list, KernelType.Fused, tile_configs)
        generator.generate_source_code()

        print("🔨 Step 3: Compiling with NO Python dependencies...")
        os.chdir(f"{CUR_DIR}/mxmoe/kernels/src/")
        
        # 使用已经替换的CMakeLists.txt (无Python依赖，无网络下载!)
        print("   Using modified CMakeLists.txt (no Python interface, no network downloads!)")
        
        os.system("cmake -B build -G Ninja")
        os.system("cmake --build build --target test")

        # 3. bench kernel 
        print("⚡ Step 4: Running pure C++ benchmark...")
        bench_save_suffix = workload_suffix.replace(".json", "")
        bench_save = f"{CUR_DIR}/out/bench/{model_id}-{dataset}-{bs}{bench_save_suffix}"
        
        # 确保输出目录存在
        os.makedirs(f"{CUR_DIR}/out/bench/", exist_ok=True)
        
        handle = subprocess.Popen(
            (
            f"./build/test bench"
            f" --input {workload_path}"
            f" --output {bench_save}"
            ),
            shell=True,
            env=os.environ.copy(),
        )
        handle.wait()
        
        print(f"✅ Layer {layer} completed! Results saved to: {bench_save}")

    print("🎉 All layers completed successfully!")
    print("📈 Pure C++/CUDA benchmark completed without Python interface dependencies!")
