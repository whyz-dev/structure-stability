"""
하이퍼파라미터 설정을 위한 환경 진단 스크립트
실행: python check_env.py
"""

import os
import sys
import platform
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from logger_config import get_check_env_logger

logger = get_check_env_logger()

# ────────────────────────────────────────────────────────
# 1. Python / OS
# ────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("[ 1. 기본 환경 ]")
logger.info(f"  Python       : {sys.version.split()[0]}")
logger.info(f"  OS           : {platform.system()} {platform.release()}")
logger.info(f"  Architecture : {platform.machine()}")

# ────────────────────────────────────────────────────────
# 2. CPU
# ────────────────────────────────────────────────────────
logger.info("[ 2. CPU ]")
cpu_count_logical  = os.cpu_count()
try:
    import psutil
    cpu_count_physical = psutil.cpu_count(logical=False)
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    logger.info(f"  물리 코어 수    : {cpu_count_physical}")
    logger.info(f"  논리 코어 수    : {cpu_count_logical}  ← NUM_WORKERS 상한선")
    logger.info(f"  권장 NUM_WORKERS: {cpu_count_logical // 2} ~ {cpu_count_logical}")
    logger.info(f"  RAM             : {ram_gb:.1f} GB")
except ImportError:
    logger.info(f"  논리 코어 수    : {cpu_count_logical}  ← NUM_WORKERS 상한선")
    logger.info(f"  권장 NUM_WORKERS: {cpu_count_logical // 2} ~ {cpu_count_logical}")
    logger.warning("  psutil 없음 - RAM 정보 생략 (pip install psutil 권장)")

# ────────────────────────────────────────────────────────
# 3. CUDA / GPU
# ────────────────────────────────────────────────────────
logger.info("[ 3. GPU / CUDA ]")
try:
    import torch
    logger.info(f"  PyTorch        : {torch.__version__}")
    logger.info(f"  CUDA available : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"  CUDA version   : {torch.version.cuda}")
        logger.info(f"  cuDNN version  : {torch.backends.cudnn.version()}")
        logger.info(f"  GPU 개수       : {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            total_mem = prop.total_memory / (1024 ** 3)
            logger.info(f"  [GPU {i}] {prop.name}")
            logger.info(f"    VRAM              : {total_mem:.1f} GB")
            logger.info(f"    SM (CUDA cores)   : {prop.multi_processor_count}")
            logger.info(f"    Compute Capability: {prop.major}.{prop.minor}")

            if   total_mem >= 40: bs_hint = "256 ~ 512"
            elif total_mem >= 24: bs_hint = "128 ~ 256"
            elif total_mem >= 16: bs_hint = "64 ~ 128"
            elif total_mem >= 10: bs_hint = "32 ~ 64"
            elif total_mem >= 6:  bs_hint = "16 ~ 32"
            else:                 bs_hint = "8 ~ 16"
            logger.info(f"    권장 BATCH_SIZE   : {bs_hint}  (IMG_SIZE·모델에 따라 조정)")

        cur      = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0)  / (1024 ** 3)
        logger.info(f"  현재 GPU 메모리 사용: {cur:.2f} GB allocated / {reserved:.2f} GB reserved")

    else:
        logger.warning("  GPU 없음 → CPU 학습 모드")
        logger.warning("  권장 BATCH_SIZE: 8 ~ 16 (CPU는 작게)")

except ImportError:
    logger.error("  PyTorch 미설치")

# ────────────────────────────────────────────────────────
# 4. 주요 라이브러리 버전
# ────────────────────────────────────────────────────────
logger.info("[ 4. 주요 라이브러리 버전 ]")
libs = {
    "torch"         : "torch",
    "torchvision"   : "torchvision",
    "timm"          : "timm",
    "numpy"         : "numpy",
    "pandas"        : "pandas",
    "cv2"           : "cv2",
    "PIL"           : "PIL",
    "sklearn"       : "sklearn",
    "albumentations": "albumentations",
    "kornia"        : "kornia",
}
for name, module in libs.items():
    try:
        m   = __import__(module)
        ver = getattr(m, "__version__", "version 확인 불가")
        logger.info(f"  {name:<18}: {ver}")
    except ImportError:
        logger.warning(f"  {name:<18}: 미설치")

# ────────────────────────────────────────────────────────
# 5. 스토리지
# ────────────────────────────────────────────────────────
logger.info("[ 5. 스토리지 ]")
try:
    import psutil
    disk = psutil.disk_usage('/')
    logger.info(f"  전체 용량  : {disk.total / (1024**3):.1f} GB")
    logger.info(f"  사용 중    : {disk.used  / (1024**3):.1f} GB")
    logger.info(f"  남은 용량  : {disk.free  / (1024**3):.1f} GB")
except Exception:
    logger.warning("  psutil 없음 - 생략")

# ────────────────────────────────────────────────────────
# 6. 권장 CFG 요약
# ────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("[ 권장 CFG 요약 ]")
try:
    import torch, os, psutil
    nw = os.cpu_count() // 2
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if   vram >= 40: bs = 256
        elif vram >= 24: bs = 128
        elif vram >= 16: bs = 64
        elif vram >= 10: bs = 32
        else:            bs = 16
    else:
        bs = 8

    logger.info(
        f"\n  CFG = {{\n"
        f"      'BATCH_SIZE'  : {bs},\t# VRAM 기준 추정값\n"
        f"      'NUM_WORKERS' : {nw},\t# 논리 코어의 절반\n"
        f"      'PIN_MEMORY'  : {torch.cuda.is_available()},\t# GPU 사용 시 True\n"
        f"  }}"
    )
except Exception:
    logger.warning("  torch/psutil 없어 CFG 요약 생략")

logger.info("=" * 60)
logger.info("※ 위 수치는 추정값입니다. nvidia-smi 로 최종 확인하세요.")
logger.info("  watch -n 0.5 nvidia-smi")