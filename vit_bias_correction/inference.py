# inference.py
import torch
import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy import interpolate
import os


def load_trained_model(checkpoint_path, model_class, device='cuda:0'):
    """학습된 모델 로드"""
    print(f"📥 Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Val Loss: {checkpoint['val_loss']:.6f}")
    
    return checkpoint


def prepare_inference_quantile_maps_gpu(
    lr_data,
    n_quantiles,
    lr_mean,
    lr_std,
    device='cuda:0',
    batch_size=2000
):
    """
    GPU 가속 추론용 분위수 맵 생성
    
    Parameters:
    -----------
    lr_data : xarray.DataArray
        예측할 LR 데이터 [T, lat, lon]
    n_quantiles : int
        분위수 개수
    lr_mean, lr_std : float
        훈련 시 사용한 정규화 통계
    device : str
        GPU 디바이스
    batch_size : int
        배치 크기
    
    Returns:
    --------
    lr_q_normalized : numpy.ndarray
        정규화된 분위수 맵 [n_quantiles, lat, lon]
    """
    print(f"\n🔄 Computing quantile maps for inference (GPU)...")
    print(f"   Data shape: {lr_data.shape}")
    print(f"   Device: {device}")
    
    T, H, W = lr_data.shape
    quantiles = np.linspace(0, 100, n_quantiles)
    
    # 분위수 레벨을 GPU로
    q_levels = torch.linspace(0, 1, n_quantiles, device=device)
    
    # 결과 배열
    lr_q_maps = np.zeros((n_quantiles, H, W), dtype=np.float32)
    
    # 데이터 reshape [T, H*W]
    data_2d = lr_data.values.reshape(T, H * W)
    
    n_pixels = H * W
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_pixels, batch_size), desc="Computing quantiles (GPU)"):
            end_idx = min(start_idx + batch_size, n_pixels)
            
            # 배치 추출
            batch_data = data_2d[:, start_idx:end_idx]
            
            # 각 픽셀별로 처리
            for local_idx in range(batch_data.shape[1]):
                pixel_data = batch_data[:, local_idx]
                valid = ~np.isnan(pixel_data)
                
                if valid.sum() >= 10:
                    # GPU로 이동 및 분위수 계산
                    pixel_tensor = torch.FloatTensor(pixel_data[valid]).to(device)
                    q_vals = torch.quantile(pixel_tensor, q_levels)
                    
                    # 결과 저장
                    global_idx = start_idx + local_idx
                    lat_idx = global_idx // W
                    lon_idx = global_idx % W
                    
                    lr_q_maps[:, lat_idx, lon_idx] = q_vals.cpu().numpy()
                else:
                    # 유효 데이터 부족
                    global_idx = start_idx + local_idx
                    lat_idx = global_idx // W
                    lon_idx = global_idx % W
                    lr_q_maps[:, lat_idx, lon_idx] = np.nan
    
    # 정규화 (훈련 시와 동일)
    lr_q_normalized = (lr_q_maps - lr_mean) / lr_std
    lr_q_normalized = np.nan_to_num(lr_q_normalized, 0)
    
    print(f"✅ Quantile maps ready: {lr_q_normalized.shape}")
    return lr_q_normalized, quantiles


def predict_quantile_maps(model, lr_q_normalized, device, batch_size=16):
    """
    모델로 HR 분위수 맵 예측
    
    Parameters:
    -----------
    model : torch.nn.Module
        학습된 모델
    lr_q_normalized : numpy.ndarray
        정규화된 LR 분위수 맵 [n_quantiles, lat, lon]
    device : str
        디바이스
    batch_size : int
        배치 크기
    
    Returns:
    --------
    hr_q_predicted : numpy.ndarray
        예측된 HR 분위수 맵
    """
    print(f"\n🤖 Predicting with model...")
    
    model.eval()
    hr_q_predicted = []
    
    n_quantiles = len(lr_q_normalized)
    
    with torch.no_grad():
        for i in tqdm(range(0, n_quantiles, batch_size), desc="Predicting"):
            batch = lr_q_normalized[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).unsqueeze(1).to(device)
            
            with torch.cuda.amp.autocast():
                pred = model(batch_tensor)
            
            hr_q_predicted.append(pred.cpu().squeeze(1).numpy())
    
    hr_q_predicted = np.concatenate(hr_q_predicted, axis=0)
    
    print(f"✅ Prediction done: {hr_q_predicted.shape}")
    return hr_q_predicted


def apply_quantile_correction_gpu(
    lr_data,
    hr_q_predicted,
    quantiles,
    hr_mean,
    hr_std,
    device='cuda:0',
    batch_size=1000
):
    """
    GPU 가속 분위수 보정 적용
    
    Parameters:
    -----------
    lr_data : xarray.DataArray
        원본 LR 데이터 [T, lat, lon]
    hr_q_predicted : numpy.ndarray
        예측된 HR 분위수 맵 (정규화됨)
    quantiles : numpy.ndarray
        분위수 배열
    hr_mean, hr_std : float
        HR 정규화 통계 (역정규화용)
    device : str
        GPU 디바이스
    batch_size : int
        배치 크기
    
    Returns:
    --------
    lr_corrected : xarray.DataArray
        보정된 데이터
    """
    print(f"\n🔧 Applying bias correction (GPU)...")
    
    # HR 분위수 맵 역정규화
    hr_q_maps = hr_q_predicted * hr_std + hr_mean
    
    # 보정된 데이터 초기화
    lr_corrected = lr_data.copy(deep=True)
    
    T, H, W = lr_data.shape
    n_pixels = H * W
    
    # 데이터 reshape
    data_2d = lr_data.values.reshape(T, H * W)
    corrected_2d = np.zeros_like(data_2d)
    
    # GPU 텐서로 변환
    q_levels = torch.FloatTensor(quantiles / 100.0).to(device)  # 0~1 범위
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_pixels, batch_size), desc="Correcting (GPU)"):
            end_idx = min(start_idx + batch_size, n_pixels)
            
            # 배치 처리
            for local_idx in range(end_idx - start_idx):
                global_idx = start_idx + local_idx
                lat_idx = global_idx // W
                lon_idx = global_idx % W
                
                lr_vals = data_2d[:, global_idx]
                hr_q = hr_q_maps[:, lat_idx, lon_idx]
                
                valid = ~np.isnan(lr_vals)
                if valid.sum() < 10:
                    continue
                
                # LR 분위수 계산 (GPU)
                lr_vals_gpu = torch.FloatTensor(lr_vals[valid]).to(device)
                lr_q_gpu = torch.quantile(lr_vals_gpu, q_levels)
                
                # HR 분위수 (GPU)
                hr_q_gpu = torch.FloatTensor(hr_q).to(device)
                
                # 보간 (CPU가 더 빠를 수 있음)
                lr_q_cpu = lr_q_gpu.cpu().numpy()
                hr_q_cpu = hr_q_gpu.cpu().numpy()
                
                # 보간 함수 생성
                transfer = interpolate.interp1d(
                    lr_q_cpu, hr_q_cpu,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(hr_q_cpu[0], hr_q_cpu[-1])
                )
                
                # 보정 적용
                corrected_2d[:, global_idx] = transfer(lr_vals)
    
    # Reshape back
    lr_corrected.values = corrected_2d.reshape(T, H, W)
    
    print(f"✅ Bias correction complete!")
    return lr_corrected


def run_bias_correction(
    lr_data_path,
    checkpoint_path,
    output_path,
    model_class,
    model_config,
    n_quantiles=100,
    device='cuda:1',
    batch_size=16,
    restore_extremes=False,
    use_gpu_quantiles=True,  # ⭐ GPU 분위수 계산 옵션
    quantile_batch_size=2000,  # ⭐ 분위수 계산 배치 크기
    var_name=None, 
    quantile_map_path=None # pkl 파일 경로
):
    """
    전체 편이보정 파이프라인 (GPU 가속)
    
    Parameters:
    -----------
    lr_data_path : str
        입력 LR 데이터 경로 (.nc)
    checkpoint_path : str
        모델 체크포인트 경로
    output_path : str
        출력 파일 경로
    model_class : class
        모델 클래스
    model_config : dict
        모델 설정
    n_quantiles : int
        분위수 개수
    device : str
        디바이스
    batch_size : int
        모델 추론 배치 크기
    restore_extremes : bool
        극값 복원 여부
    use_gpu_quantiles : bool
        분위수 계산에 GPU 사용 여부
    quantile_batch_size : int
        분위수 계산 배치 크기
    """
    
    print("="*60)
    print("🚀 Starting Bias Correction Pipeline (GPU Accelerated)")
    print("="*60)
    
    # 1. 체크포인트 로드
    checkpoint = load_trained_model(checkpoint_path, model_class, device)
    
    lr_mean, lr_std = checkpoint['lr_stats']
    hr_mean, hr_std = checkpoint['hr_stats']
    
    print(f"\n📊 Normalization stats:")
    print(f"   LR: mean={lr_mean:.4f}, std={lr_std:.4f}")
    print(f"   HR: mean={hr_mean:.4f}, std={hr_std:.4f}")
    
    # 2. 모델 초기화
    print(f"\n🔧 Initializing model...")
    model = model_class(**model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 3. LR 데이터 로드
    print(f"\n📂 Loading LR data from: {lr_data_path}")
    lr_ds = xr.open_dataset(lr_data_path)
    
    # 변수 이름 자동 감지
    if var_name is None:
        var_candidates = [var for var in lr_ds.data_vars if lr_ds[var].ndim == 3]
        if not var_candidates:
            raise ValueError("No suitable 3D variable found in the dataset.")
        var_name = var_candidates[0]
        print(f"   Detected variable: {var_name}")
    lr_data = lr_ds[var_name]
    
    print(f"   Variable: {var_name}")
    print(f"   Shape: {lr_data.shape}")
    print(f"   Time range: {lr_data.time.values[0]} ~ {lr_data.time.values[-1]}")
    if quantile_map_path is not None and os.path.exists(quantile_map_path):
        print(f"   Loading precomputed quantile maps from: {quantile_map_path}")
        import pickle
        with open(quantile_map_path, 'rb') as f:
            quantile_data = pickle.load(f)
        lr_q_normalized = quantile_data['train'][0]
        quantiles = quantile_data['quantiles']
    else:
        print(f"   No precomputed quantile maps found.")
        print(f"   Computing quantile maps...")
        # 4. 분위수 맵 생성 (GPU 가속)
        if use_gpu_quantiles:
            lr_q_normalized, quantiles = prepare_inference_quantile_maps_gpu(
                lr_data, n_quantiles, lr_mean, lr_std,
                device=device, batch_size=quantile_batch_size
            )
        else:
            # CPU 버전 (기존 코드)
            from inference import prepare_inference_quantile_maps
            lr_q_normalized, quantiles = prepare_inference_quantile_maps(
                lr_data, n_quantiles, lr_mean, lr_std
            )
    
    # 5. 모델 예측
    hr_q_predicted = predict_quantile_maps(
        model, lr_q_normalized, device, batch_size
    )
    
    # 6. 편이보정 적용 (GPU 가속)
    if use_gpu_quantiles:
        lr_corrected = apply_quantile_correction_gpu(
            lr_data, hr_q_predicted, quantiles, hr_mean, hr_std,
            device=device, batch_size=quantile_batch_size
        )
    else:
        # CPU 버전
        from inference import apply_quantile_correction
        lr_corrected = apply_quantile_correction(
            lr_data, hr_q_predicted, quantiles, hr_mean, hr_std
        )
    
    # 7. 극값 복원 (선택)
    if restore_extremes:
        from inference import restore_extreme_values
        target_stats = {
            'mean': hr_mean,
            'std': hr_std,
            'q01': hr_mean - 3 * hr_std,
            'q99': hr_mean + 3 * hr_std
        }
        lr_corrected = restore_extreme_values(lr_corrected, target_stats)
    
    # 8. 저장
    print(f"\n💾 Saving corrected data to: {output_path}")
    
    # 메타데이터 추가
    lr_corrected.attrs['bias_correction'] = 'Applied'
    lr_corrected.attrs['model'] = model_class.__name__
    lr_corrected.attrs['checkpoint'] = checkpoint_path
    lr_corrected.attrs['n_quantiles'] = n_quantiles
    
    # Dataset으로 변환 후 저장
    output_ds = lr_corrected.to_dataset(name=var_name + '_corrected')
    output_ds.to_netcdf(output_path)
    
    print(f"✅ Saved successfully!")
    
    # 9. 통계 비교
    print(f"\n📊 Statistics Comparison:")
    print(f"{'Metric':<15} {'Original':<15} {'Corrected':<15}")
    print(f"{'-'*45}")
    print(f"{'Mean':<15} {float(lr_data.mean()):<15.4f} {float(lr_corrected.mean()):<15.4f}")
    print(f"{'Std':<15} {float(lr_data.std()):<15.4f} {float(lr_corrected.std()):<15.4f}")
    print(f"{'Min':<15} {float(lr_data.min()):<15.4f} {float(lr_corrected.min()):<15.4f}")
    print(f"{'Max':<15} {float(lr_data.max()):<15.4f} {float(lr_corrected.max()):<15.4f}")
    
    print("\n" + "="*60)
    print("🎉 Bias Correction Completed!")
    print("="*60)
    
    return lr_corrected


if __name__ == "__main__":
    # 테스트
    from model import SimpleViT
    
    model_config = {
        'img_size': 240,
        'patch_size': 8,
        'in_chans': 1,
        'out_chans': 1,
        'embed_dim': 512,
        'depth': 12,
        'num_heads': 8,
        'dropout': 0.1
    }
    
    lr_corrected = run_bias_correction(
        lr_data_path='./data/lr_monthly_max.nc',
        checkpoint_path='checkpoints/simple_vit/best_model.pth',
        output_path='./data/lr_corrected_monthly_max.nc',
        model_class=SimpleViT,
        model_config=model_config,
        n_quantiles=500,
        device='cuda:1',
        batch_size=32,
        use_gpu_quantiles=True,        # ⭐ GPU 사용
        quantile_batch_size=2000,      # ⭐ 배치 크기
        restore_extremes=False
    )