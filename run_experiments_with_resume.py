import os
import json
from main import do_run_experiment, load_config
import triton.runtime.autotuner

# 定义路径
data_path = "/root/RingTool/data/rings/"
config_dir = "/root/RingTool/config"
completed_file = "/root/RingTool/completed_configs.txt"  # 记录已完成配置
failed_configs = []  # 记录失败的配置文件

# 加载已完成的配置文件
completed_configs = set()
if os.path.exists(completed_file):
    with open(completed_file, "r") as f:
        completed_configs = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(completed_configs)} completed configs from {completed_file}")

# 遍历 supervised 文件夹下的所有 JSON 文件
config_files = [os.path.join(root, f) for root, _, files in os.walk(config_dir) for f in files if f.endswith(".json")]
config_files = [f for f in config_files if f not in completed_configs]  # 跳过已完成
print(f"Found {len(config_files)} JSON config files to process in {config_dir}")

# 批量运行实验
for config_path in config_files:
    print(f"\nProcessing: {config_path}")
    try:
        if "mamba" not in config_path:
            continue
        
        # 加载配置
        config = load_config(config_path)
        config["_config_path_"] = config_path
        config["mode"] = "5fold"

        # 第一次尝试运行
        try:
            do_run_experiment(config, data_path, False)
            print(f"Success: {config_path}")
            # 记录成功完成的配置文件
            with open(completed_file, "a") as f:
                f.write(f"{config_path}\n")

        except triton.runtime.autotuner.OutOfResources as e:
            print(f"Shared memory error for {config_path}: {str(e)}")
            # 降低 batch_size
            original_batch_size = config["dataset"]["batch_size"]
            new_batch_size = max(1, original_batch_size - 32)
            print(f"Retrying with batch_size={new_batch_size}")
            config["dataset"]["batch_size"] = new_batch_size
            config["test"]["batch_size"] = new_batch_size

            # 保存调整后的配置
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            # 重新加载并重试
            config = load_config(config_path)
            config["_config_path_"] = config_path
            try:
                do_run_experiment(config, data_path, False)
                print(f"Success after retry: {config_path}")
                # 记录成功完成的配置文件
                with open(completed_file, "a") as f:
                    f.write(f"{config_path}\n")
            except Exception as e2:
                print(f"Retry failed for {config_path}: {str(e2)}")
                failed_configs.append(config_path)
                continue

        # 检查日志
        log_files = sorted(
            [f for f in os.listdir("logs") if f.startswith("rtool-")],
            key=lambda x: os.path.getmtime(os.path.join("logs", x))
        )
        if log_files:
            with open(f"logs/{log_files[-1]}", "r") as f:
                print(f"Log for {config_path}:\n{f.read()}")
        else:
            print(f"No log file found for {config_path}")

    except Exception as e:
        print(f"Error running {config_path}: {str(e)}")
        failed_configs.append(config_path)
        continue

# 输出失败的配置文件
print("\nExperiment Summary:")
if failed_configs:
    print(f"Failed to run {len(failed_configs)} configs:")
    for config in failed_configs:
        print(f"  - {config}")
else:
    print("All configs ran successfully!")

print("\nAll experiments completed!")