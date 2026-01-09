# utils.py
import numpy as np
import torch
import pickle
import os
import glob
from tqdm import tqdm


# ============================================================================
# 1. 고정 범위 분위수 맵 생성 (GPU 가속)
# ============================================================================

def prepare_quantile_maps_fixed_range_gpu(
    data,
    n_quantiles=500,
    value_min=260,
    value_max=320,
    device='cuda:0',
    batch_size=2000
):
    """
    고정 범위 기반 분위수 맵 생성 (GPU 가속)
    
    Parameters:
    -----------
    data : xarray.DataArray
        입력 데이터 [T, lat, lon] (물리 단위, e.g., Kelvin)
    n_quantiles : int
        분위수 개수
    value_min, value_max : float
        고정 범위 (물리 단위)
    device : str
        GPU 디바이스
    batch_size : int
        배치 크기
    
    Returns:
    --------
    q_maps_norm : numpy.ndarray
        정규화된 분위수 맵 [n_quantiles, lat, lon], [0, 1] 범위
    quantiles : numpy.ndarray
        분위수 배열 [0, 100]
    """
    
    print(f"\n🔄 Computing quantile maps (GPU: {device})")
    print(f"   Data shape: {data.shape}")
    print(f"   Value range: [{value_min}, {value_max}]")
    print(f"   N quantiles: {n_quantiles}")
    
    T, H, W = data.shape
    quantiles = np.linspace(0, 100, n_quantiles)
    q_levels = torch.linspace(0, 1, n_quantiles, device=device)
    
    # 1. 정규화 [0, 1]
    data_norm = (data.values - value_min) / (value_max - value_min)
    data_norm = np.clip(data_norm, 0, 1)  # ⭐ 범위 클리핑
    
    # 범위 밖 데이터 비율
    outside_pct = ((data.values < value_min) | (data.values > value_max)).sum() / data.size * 100
    if outside_pct > 0:
        print(f"   ⚠️  {outside_pct:.2f}% of data outside range (clipped)")
    
    # 2. 분위수 맵 계산
    q_maps = np.zeros((n_quantiles, H, W), dtype=np.float32)
    data_2d = data_norm.reshape(T, H * W)
    n_pixels = H * W
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_pixels, batch_size), desc="  Computing"):
            end_idx = min(start_idx + batch_size, n_pixels)
            
            for local_idx in range(end_idx - start_idx):
                global_idx = start_idx + local_idx
                lat_idx = global_idx // W
                lon_idx = global_idx % W
                
                pixel_data = data_2d[:, global_idx]
                valid = ~np.isnan(pixel_data)
                
                if valid.sum() >= 10:
                    pixel_tensor = torch.FloatTensor(pixel_data[valid]).to(device)
                    q_vals = torch.quantile(pixel_tensor, q_levels)
                    q_maps[:, lat_idx, lon_idx] = q_vals.cpu().numpy()
                else:
                    # 유효 데이터 부족 → 균등 분포
                    q_maps[:, lat_idx, lon_idx] = q_levels.cpu().numpy()
    
    print(f"   ✅ Quantile map: [{q_maps.min():.3f}, {q_maps.max():.3f}]")
    
    return q_maps, quantiles


def prepare_all_quantile_maps_fixed_range(
    lr_train, hr_train,
    lr_val, hr_val,
    n_quantiles=500,
    lr_range=(260, 320),
    hr_range=(260, 320),
    cache_path=None,
    force_recompute=False,
    device='cuda:0',
    batch_size=2000
):
    """
    전체 분위수 맵 생성 (고정 범위, 캐싱 지원)
    
    Parameters:
    -----------
    lr_train, hr_train : xarray.DataArray
        훈련 데이터 (물리 단위)
    lr_val, hr_val : xarray.DataArray
        검증 데이터 (물리 단위)
    n_quantiles : int
        분위수 개수
    lr_range, hr_range : tuple
        고정 범위 (min, max)
    cache_path : str or None
        캐시 파일 경로
    force_recompute : bool
        강제 재계산
    device : str
        GPU 디바이스
    batch_size : int
        배치 크기
    
    Returns:
    --------
    result : dict
        {'train': (lr_q, hr_q), 'val': (lr_q, hr_q), 'quantiles': ..., 'normalization': ...}
    """
    
    # 캐시 확인
    if cache_path and os.path.exists(cache_path) and not force_recompute:
        print(f"\n📂 Loading from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            result = pickle.load(f)
        print("✅ Cache loaded")
        return result
    
    print("\n" + "="*60)
    print("🚀 Computing Quantile Maps (Fixed Range)")
    print("="*60)
    
    lr_min, lr_max = lr_range
    hr_min, hr_max = hr_range
    
    print(f"\n📊 Configuration:")
    print(f"   LR range: [{lr_min}K, {lr_max}K] = [{lr_min-273.15:.1f}°C, {lr_max-273.15:.1f}°C]")
    print(f"   HR range: [{hr_min}K, {hr_max}K] = [{hr_min-273.15:.1f}°C, {hr_max-273.15:.1f}°C]")
    print(f"   N quantiles: {n_quantiles}")
    print(f"   Device: {device}")
    
    # 데이터 범위 확인
    print(f"\n📊 Data statistics:")
    print(f"   LR train: [{float(lr_train.min()):.2f}, {float(lr_train.max()):.2f}] K")
    print(f"   HR train: [{float(hr_train.min()):.2f}, {float(hr_train.max()):.2f}] K")
    print(f"   LR val:   [{float(lr_val.min()):.2f}, {float(lr_val.max()):.2f}] K")
    print(f"   HR val:   [{float(hr_val.min()):.2f}, {float(hr_val.max()):.2f}] K")
    
    # 1. LR Train
    print(f"\n[1/4] LR Train")
    lr_q_train, quantiles = prepare_quantile_maps_fixed_range_gpu(
        lr_train, n_quantiles, lr_min, lr_max, device, batch_size
    )
    
    # 2. HR Train
    print(f"\n[2/4] HR Train")
    hr_q_train, _ = prepare_quantile_maps_fixed_range_gpu(
        hr_train, n_quantiles, hr_min, hr_max, device, batch_size
    )
    
    # 3. LR Val
    print(f"\n[3/4] LR Val")
    lr_q_val, _ = prepare_quantile_maps_fixed_range_gpu(
        lr_val, n_quantiles, lr_min, lr_max, device, batch_size
    )
    
    # 4. HR Val
    print(f"\n[4/4] HR Val")
    hr_q_val, _ = prepare_quantile_maps_fixed_range_gpu(
        hr_val, n_quantiles, hr_min, hr_max, device, batch_size
    )
    
    # 결과 패키징
    result = {
        'train': (lr_q_train, hr_q_train),
        'val': (lr_q_val, hr_q_val),
        'quantiles': quantiles,
        'normalization': {
            'type': 'fixed_range',
            'lr_min': lr_min,
            'lr_max': lr_max,
            'hr_min': hr_min,
            'hr_max': hr_max
        },
        'metadata': {
            'n_quantiles': n_quantiles,
            'lr_train_shape': lr_train.shape,
            'hr_train_shape': hr_train.shape,
            'lr_val_shape': lr_val.shape,
            'hr_val_shape': hr_val.shape,
            'device': device
        }
    }
    
    # 캐시 저장
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"\n💾 Saving to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   File size: {os.path.getsize(cache_path) / 1024**2:.2f} MB")
    
    print(f"\n✅ Quantile maps ready!")
    print(f"   Train: LR={lr_q_train.shape}, HR={hr_q_train.shape}")
    print(f"   Val:   LR={lr_q_val.shape}, HR={hr_q_val.shape}")
    print("="*60 + "\n")
    
    return result


# ============================================================================
# 2. 캐시 관리
# ============================================================================

def inspect_quantile_cache(cache_path):
    """
    분위수 맵 캐시 정보 확인
    
    Parameters:
    -----------
    cache_path : str
        캐시 파일 경로
    """
    if not os.path.exists(cache_path):
        print(f"❌ Cache file not found: {cache_path}")
        return
    
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    
    print("\n" + "="*60)
    print(f"Quantile Cache Info")
    print("="*60)
    print(f"File: {cache_path}")
    print(f"Size: {os.path.getsize(cache_path) / 1024**2:.2f} MB")
    print()
    
    # 분위수 맵
    lr_q_train, hr_q_train = data['train']
    lr_q_val, hr_q_val = data['val']
    
    print(f"Shapes:")
    print(f"  LR train: {lr_q_train.shape}")
    print(f"  HR train: {hr_q_train.shape}")
    print(f"  LR val:   {lr_q_val.shape}")
    print(f"  HR val:   {hr_q_val.shape}")
    print()
    
    print(f"Quantiles: {len(data['quantiles'])} levels")
    print(f"  Range: [{data['quantiles'][0]:.1f}, {data['quantiles'][-1]:.1f}]")
    print()
    
    # 정규화 정보
    if 'normalization' in data:
        norm = data['normalization']
        print(f"Normalization:")
        print(f"  Type: {norm['type']}")
        if norm['type'] == 'fixed_range':
            print(f"  LR: [{norm['lr_min']}K, {norm['lr_max']}K]")
            print(f"  HR: [{norm['hr_min']}K, {norm['hr_max']}K]")
    else:
        print("Normalization: Not found (old format)")
    print()
    
    # 값 범위
    print(f"Value ranges (normalized):")
    print(f"  LR train: [{lr_q_train.min():.3f}, {lr_q_train.max():.3f}]")
    print(f"  HR train: [{hr_q_train.min():.3f}, {hr_q_train.max():.3f}]")
    print(f"  LR val:   [{lr_q_val.min():.3f}, {lr_q_val.max():.3f}]")
    print(f"  HR val:   [{hr_q_val.min():.3f}, {hr_q_val.max():.3f}]")
    print()
    
    # 메타데이터
    if 'metadata' in data:
        meta = data['metadata']
        print(f"Metadata:")
        for key, value in meta.items():
            print(f"  {key}: {value}")
    
    print("="*60 + "\n")


def list_all_caches(cache_dir='cache'):
    """
    모든 캐시 파일 목록
    
    Parameters:
    -----------
    cache_dir : str
        캐시 디렉토리
    """
    cache_files = glob.glob(os.path.join(cache_dir, '*.pkl'))
    
    if not cache_files:
        print(f"\nNo cache files in {cache_dir}")
        return
    
    print("\n" + "="*80)
    print(f"Cache Files in {cache_dir}")
    print("="*80)
    
    total_size = 0
    for cache_file in sorted(cache_files):
        size_mb = os.path.getsize(cache_file) / 1024**2
        total_size += size_mb
        print(f"{os.path.basename(cache_file):<50} {size_mb:>10.2f} MB")
    
    print("="*80)
    print(f"{'Total':<50} {total_size:>10.2f} MB")
    print("="*80 + "\n")


def delete_cache(cache_path):
    """
    캐시 파일 삭제
    
    Parameters:
    -----------
    cache_path : str
        삭제할 캐시 파일 경로
    """
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"🗑️  Deleted: {cache_path}")
    else:
        print(f"❌ Not found: {cache_path}")


def clear_all_caches(cache_dir='cache', confirm=True):
    """
    모든 캐시 삭제
    
    Parameters:
    -----------
    cache_dir : str
        캐시 디렉토리
    confirm : bool
        확인 여부
    """
    cache_files = glob.glob(os.path.join(cache_dir, '*.pkl'))
    
    if not cache_files:
        print(f"\nNo cache files in {cache_dir}")
        return
    
    if confirm:
        list_all_caches(cache_dir)
        response = input(f"Delete {len(cache_files)} cache files? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled")
            return
    
    for cache_file in cache_files:
        os.remove(cache_file)
        print(f"🗑️  Deleted: {os.path.basename(cache_file)}")
    
    print(f"\n✅ Deleted {len(cache_files)} files")


# ============================================================================
# 3. GPU 메모리 관리
# ============================================================================

def print_gpu_memory(device_id=None):
    """
    GPU 메모리 사용량
    
    Parameters:
    -----------
    device_id : int or None
        특정 GPU (None이면 모든 GPU)
    """
    import torch
    
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available\n")
        return
    
    if device_id is not None:
        devices = [device_id]
    else:
        devices = range(torch.cuda.device_count())
    
    print("\n" + "="*60)
    print("GPU Memory Usage")
    print("="*60)
    
    for i in devices:
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total:     {total:.2f} GB")
        print(f"  Allocated: {allocated:.2f} GB ({allocated/total*100:.1f}%)")
        print(f"  Reserved:  {reserved:.2f} GB ({reserved/total*100:.1f}%)")
        print(f"  Free:      {total - allocated:.2f} GB")
    
    print("="*60 + "\n")


def clear_gpu_memory(device_id=None):
    """
    GPU 메모리 정리
    
    Parameters:
    -----------
    device_id : int or None
        특정 GPU (None이면 모든 GPU)
    """
    import torch
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print("\n🧹 Clearing GPU memory...")
    
    if device_id is not None:
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"✅ Cleared GPU {device_id}")
    else:
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        print(f"✅ Cleared all {torch.cuda.device_count()} GPUs")
    
    print_gpu_memory(device_id)


# ============================================================================
# 4. CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            cache_dir = sys.argv[2] if len(sys.argv) > 2 else 'cache'
            list_all_caches(cache_dir)
        
        elif command == "inspect":
            if len(sys.argv) < 3:
                print("Usage: python utils.py inspect <cache_file>")
            else:
                inspect_quantile_cache(sys.argv[2])
        
        elif command == "delete":
            if len(sys.argv) < 3:
                print("Usage: python utils.py delete <cache_file>")
            else:
                delete_cache(sys.argv[2])
        
        elif command == "clear":
            cache_dir = sys.argv[2] if len(sys.argv) > 2 else 'cache'
            clear_all_caches(cache_dir, confirm=True)
        
        elif command == "gpu":
            device_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
            print_gpu_memory(device_id)
        
        elif command == "clear-gpu":
            device_id = int(sys.argv[2]) if len(sys.argv) > 2 else None
            clear_gpu_memory(device_id)
        
        else:
            print(f"Unknown command: {command}")
            print("\nAvailable commands:")
            print("  list [cache_dir]         - List all cache files")
            print("  inspect <file>           - Inspect cache file")
            print("  delete <file>            - Delete cache file")
            print("  clear [cache_dir]        - Clear all caches")
            print("  gpu [device_id]          - Show GPU memory")
            print("  clear-gpu [device_id]    - Clear GPU memory")
    
    else:
        print("\n" + "="*60)
        print("Utils - Cache & GPU Management")
        print("="*60)
        print("\nUsage: python utils.py <command> [args]")
        print("\nCommands:")
        print("  list [cache_dir]         - List all cache files")
        print("  inspect <file>           - Inspect cache file details")
        print("  delete <file>            - Delete specific cache file")
        print("  clear [cache_dir]        - Clear all cache files")
        print("  gpu [device_id]          - Show GPU memory usage")
        print("  clear-gpu [device_id]    - Clear GPU memory cache")
        print("\nExamples:")
        print("  python utils.py list")
        print("  python utils.py inspect cache/quantile_maps.pkl")
        print("  python utils.py clear cache")
        print("  python utils.py gpu 0")
        print("  python utils.py clear-gpu")
        print("="*60 + "\n")