from __future__ import annotations

from pathlib import Path
from huggingface_hub import hf_hub_download


# 设定模型所在的 repo 和模型类型
REPO_ID = "zhemaxiya/marine-image-api-models"
REPO_TYPE = "model"

#TODO：需要修改为你的目标下载地址
# 设定本地存储目录
BASE_DIR = Path("./downloaded_assets")

# 定义你需要下载的文件及其在 repo 中的路径
FILES = {
    "router_model_path": ("router/best.pt", "router/best.pt"),
    "sonar_cls_path": ("sonar/best.pt", "sonar/best.pt"),
    "fish_coral_cls_path": ("fish_coral_cls/best.pt", "fish_coral_cls/best.pt"),
    "fish_model_path": ("fish_detector/best.pt", "fish_detector/best.pt"),
    "coral_model_path": ("coral_detector/best.pt", "coral_detector/best.pt"),
    "bioclip2_checkpoint": ("bioclip2/epoch_50.pt", "bioclip2/epoch_50.pt"),
    "bioclip2_terms_path": ("bioclip2/terms.txt", "bioclip2/terms.txt"),
}


def main() -> None:
    # 创建存储目录
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # 下载的文件存储在字典里
    downloaded = {}

    for key, (repo_file, local_relpath) in FILES.items():
        # 创建本地的子目录结构
        local_dir = BASE_DIR / Path(local_relpath).parent
        local_dir.mkdir(parents=True, exist_ok=True)

        # 下载文件并保存到本地
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_file,
            repo_type=REPO_TYPE,
            local_dir=str(BASE_DIR),
            local_dir_use_symlinks=False,
        )
        downloaded[key] = local_path
        print(f"[OK] {key}: {local_path}")

    # 输出建议的环境变量设置
    print("\nSuggested environment variables:")
    for key, path in downloaded.items():
        env_name = key.upper()
        print(f'export {env_name}="{path}"')


if __name__ == "__main__":
    main()
