# main.py
"""
Complete Bias Correction Pipeline (Fixed Range)

Workflow:
1. Data Preparation - Historical 분위수 맵 생성
2. Model Training - ViT 학습
3. Inference - SSP 시나리오 보정
"""

import os
import argparse
import xarray as xr
from pathlib import Path

# 우리 모듈들
from utils import (
    prepare_all_quantile_maps_fixed_range,
    inspect_quantile_cache,
    print_gpu_memory,
    clear_gpu_memory
)
from train import train_fixed_range, get_default_config
from inference2 import run_bias_correction_fixed_range, run_multiple_scenarios
from model import SimpleViT


def setup_directories(base_dir='./'):
    """필요한 디렉토리 생성"""
    dirs = {
        'cache': os.path.join(base_dir, 'cache'),
        'checkpoints': os.path.join(base_dir, 'checkpoints/vit_fixed_range'),
        'plots': os.path.join(base_dir, 'plots/vit_fixed_range'),
        'outputs': os.path.join(base_dir, 'outputs')
    }
    
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"✅ {name:12s}: {path}")
    
    return dirs


def step1_prepare_data(
    gcm_hist_path,
    obs_hist_path,
    dirs,
    n_quantiles=500,
    lr_range=(260, 320),
    hr_range=(260, 320),
    device='cuda:0',
    force_recompute=False
):
    """
    Step 1: 데이터 준비 - Historical 분위수 맵 생성
    
    Parameters:
    -----------
    gcm_hist_path : str
        Historical GCM 데이터 경로
    obs_hist_path : str
        Historical OBS 데이터 경로
    dirs : dict
        디렉토리 정보
    n_quantiles : int
        분위수 개수
    lr_range, hr_range : tuple
        고정 범위 (min, max)
    device : str
        GPU 디바이스
    force_recompute : bool
        강제 재계산
    
    Returns:
    --------
    cache_path : str
        생성된 캐시 파일 경로
    """
    
    print("\n" + "="*80)
    print("STEP 1: DATA PREPARATION")
    print("="*80)
    
    # 캐시 경로
    cache_filename = f'quantile_maps_q{n_quantiles}_{lr_range[0]}_{lr_range[1]}.pkl'
    cache_path = os.path.join(dirs['cache'], cache_filename)
    
    # 캐시 확인
    if os.path.exists(cache_path) and not force_recompute:
        print(f"\n✅ Cache exists: {cache_path}")
        inspect_quantile_cache(cache_path)
        return cache_path
    
    # 데이터 로드
    print(f"\n📂 Loading data...")
    print(f"   GCM Historical: {gcm_hist_path}")
    print(f"   OBS Historical: {obs_hist_path}")
    
    gcm_hist = xr.open_dataset(gcm_hist_path)
    obs_hist = xr.open_dataset(obs_hist_path)
    
    # 변수 이름 감지
    gcm_var = [v for v in gcm_hist.data_vars if gcm_hist[v].ndim == 3][0]
    obs_var = [v for v in obs_hist.data_vars if obs_hist[v].ndim == 3][0]
    
    print(f"   GCM variable: {gcm_var}")
    print(f"   OBS variable: {obs_var}")
    
    # Train/Val 분할 (80/20)
    print(f"\n✂️  Splitting train/val (80/20)...")
    
    gcm_times = gcm_hist.time.values
    obs_times = obs_hist.time.values
    
    split_idx_gcm = int(len(gcm_times) * 0.8)
    split_idx_obs = int(len(obs_times) * 0.8)
    
    lr_train = gcm_hist[gcm_var].isel(time=slice(0, split_idx_gcm))
    lr_val = gcm_hist[gcm_var].isel(time=slice(split_idx_gcm, None))
    hr_train = obs_hist[obs_var].isel(time=slice(0, split_idx_obs))
    hr_val = obs_hist[obs_var].isel(time=slice(split_idx_obs, None))
    
    print(f"   LR train: {lr_train.shape} ({lr_train.time.values[0]} ~ {lr_train.time.values[-1]})")
    print(f"   LR val:   {lr_val.shape} ({lr_val.time.values[0]} ~ {lr_val.time.values[-1]})")
    print(f"   HR train: {hr_train.shape} ({hr_train.time.values[0]} ~ {hr_train.time.values[-1]})")
    print(f"   HR val:   {hr_val.shape} ({hr_val.time.values[0]} ~ {hr_val.time.values[-1]})")
    
    # 분위수 맵 생성
    print(f"\n🚀 Computing quantile maps...")
    result = prepare_all_quantile_maps_fixed_range(
        lr_train, hr_train,
        lr_val, hr_val,
        n_quantiles=n_quantiles,
        lr_range=lr_range,
        hr_range=hr_range,
        cache_path=cache_path,
        force_recompute=force_recompute,
        device=device,
        batch_size=2000
    )
    
    print(f"\n✅ Data preparation complete!")
    print(f"   Cache saved: {cache_path}")
    
    # 캐시 검사
    inspect_quantile_cache(cache_path)
    
    return cache_path


def step2_train_model(
    cache_path,
    dirs,
    config=None,
    device='cuda:1',
    resume_from=None
):
    """
    Step 2: 모델 학습
    
    Parameters:
    -----------
    cache_path : str
        분위수 맵 캐시 경로
    dirs : dict
        디렉토리 정보
    config : dict or None
        학습 설정 (None이면 기본값)
    device : str
        GPU 디바이스
    resume_from : str or None
        재개할 체크포인트 경로
    
    Returns:
    --------
    checkpoint_path : str
        학습된 모델 체크포인트 경로
    """
    
    print("\n" + "="*80)
    print("STEP 2: MODEL TRAINING")
    print("="*80)
    
    # 데이터 로드
    print(f"\n📂 Loading quantile maps from: {cache_path}")
    import pickle
    with open(cache_path, 'rb') as f:
        qdata = pickle.load(f)
    
    lr_q_train, hr_q_train = qdata['train']
    lr_q_val, hr_q_val = qdata['val']
    quantiles = qdata['quantiles']
    normalization = qdata['normalization']
    
    print(f"   Train: LR={lr_q_train.shape}, HR={hr_q_train.shape}")
    print(f"   Val:   LR={lr_q_val.shape}, HR={hr_q_val.shape}")
    print(f"   Normalization: {normalization['type']}")
    
    # 설정
    if config is None:
        config = get_default_config(use_fixed_range=True)
    
    # 디렉토리 설정
    config['checkpoint_dir'] = dirs['checkpoints']
    config['plot_dir'] = dirs['plots']
    config['device'] = device
    
    print(f"\n⚙️  Training configuration:")
    for key, value in config.items():
        print(f"   {key:20s}: {value}")
    
    # GPU 메모리 확인
    print_gpu_memory(int(device.split(':')[1]) if ':' in device else None)
    
    # 학습
    print(f"\n🚀 Starting training...")
    
    if resume_from:
        print(f"   Resuming from: {resume_from}")
        # TODO: resume 로직 구현
    
    trainer = train_fixed_range(
        lr_q_train, hr_q_train,
        lr_q_val, hr_q_val,
        quantiles=quantiles,
        lr_range=(normalization['lr_min'], normalization['lr_max']),
        hr_range=(normalization['hr_min'], normalization['hr_max']),
        custom_config=config
    )
    
    checkpoint_path = os.path.join(dirs['checkpoints'], 'best_model.pth')
    
    print(f"\n✅ Training complete!")
    print(f"   Best loss: {trainer.best_loss:.6f}")
    print(f"   Checkpoint: {checkpoint_path}")
    
    # GPU 메모리 정리
    clear_gpu_memory(int(device.split(':')[1]) if ':' in device else None)
    
    return checkpoint_path


def step3_inference(
    ssp_paths,
    checkpoint_path,
    cache_path,
    dirs,
    device='cuda:1',
    batch_size=32,
    var_name='tasmax'
):
    """
    Step 3: SSP 시나리오 보정
    
    Parameters:
    -----------
    ssp_paths : dict
        SSP 시나리오 파일 경로 {'ssp245': '/path/to/ssp245.nc', ...}
    checkpoint_path : str
        모델 체크포인트 경로
    cache_path : str
        분위수 맵 캐시 경로
    dirs : dict
        디렉토리 정보
    device : str
        GPU 디바이스
    batch_size : int
        배치 크기
    var_name : str
        변수 이름
    
    Returns:
    --------
    results : dict
        {scenario: corrected_data}
    """
    
    print("\n" + "="*80)
    print("STEP 3: INFERENCE (SSP SCENARIOS)")
    print("="*80)
    
    # 모델 설정
    print(f"\n⚙️  Model configuration:")
    model_config = {
        'img_size': 240,
        'patch_size': 8,
        'in_chans': 1,
        'out_chans': 1,
        'embed_dim': 512,
        'depth': 12,
        'num_heads': 8,
        'dropout': 0.1,
        'use_sigmoid': True  # ⭐ 고정 범위
    }
    
    for key, value in model_config.items():
        print(f"   {key:20s}: {value}")
    
    # GPU 메모리 확인
    print_gpu_memory(int(device.split(':')[1]) if ':' in device else None)
    
    # 각 시나리오 처리
    results = {}
    
    for i, (scenario, input_path) in enumerate(ssp_paths.items(), 1):
        print(f"\n{'='*80}")
        print(f"Processing {i}/{len(ssp_paths)}: {scenario}")
        print(f"{'='*80}")
        
        if not os.path.exists(input_path):
            print(f"⚠️  Skipping: Input file not found")
            continue
        
        output_path = os.path.join(dirs['outputs'], f'vit_{scenario}_corrected.nc')
        
        try:
            corrected = run_bias_correction_fixed_range(
                lr_data_path=input_path,
                checkpoint_path=checkpoint_path,
                quantile_map_path=cache_path,
                output_path=output_path,
                model_class=SimpleViT,
                model_config=model_config,
                device=device,
                batch_size=batch_size,
                var_name=var_name
            )
            
            results[scenario] = corrected
            print(f"✅ {scenario} completed!")
            
        except Exception as e:
            print(f"❌ {scenario} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Inference complete!")
    print(f"   Processed: {len(results)}/{len(ssp_paths)} scenarios")
    
    # GPU 메모리 정리
    clear_gpu_memory(int(device.split(':')[1]) if ':' in device else None)
    
    return results


def run_full_pipeline(
    gcm_hist_path,
    obs_hist_path,
    ssp_paths,
    base_dir='./',
    n_quantiles=500,
    lr_range=(260, 320),
    hr_range=(260, 320),
    num_epochs=200,
    data_device='cuda:0',
    train_device='cuda:1',
    inference_device='cuda:1',
    skip_data_prep=False,
    skip_training=False,
    force_recompute=False
):
    """
    전체 파이프라인 실행
    
    Parameters:
    -----------
    gcm_hist_path : str
        Historical GCM 데이터
    obs_hist_path : str
        Historical OBS 데이터
    ssp_paths : dict
        SSP 시나리오들 {'ssp245': '/path/to/ssp245.nc', ...}
    base_dir : str
        작업 디렉토리
    n_quantiles : int
        분위수 개수
    lr_range, hr_range : tuple
        고정 범위
    num_epochs : int
        학습 에폭
    data_device : str
        데이터 준비용 GPU
    train_device : str
        학습용 GPU
    inference_device : str
        추론용 GPU
    skip_data_prep : bool
        데이터 준비 건너뛰기
    skip_training : bool
        학습 건너뛰기
    force_recompute : bool
        강제 재계산
    
    Returns:
    --------
    results : dict
        전체 결과
    """
    
    print("\n" + "="*80)
    print("🚀 FULL BIAS CORRECTION PIPELINE")
    print("="*80)
    print(f"\nInput files:")
    print(f"  GCM Historical: {gcm_hist_path}")
    print(f"  OBS Historical: {obs_hist_path}")
    print(f"  SSP Scenarios:")
    for scenario, path in ssp_paths.items():
        print(f"    {scenario:10s}: {path}")
    print(f"\nConfiguration:")
    print(f"  Base directory: {base_dir}")
    print(f"  N quantiles:    {n_quantiles}")
    print(f"  LR range:       {lr_range} K")
    print(f"  HR range:       {hr_range} K")
    print(f"  Num epochs:     {num_epochs}")
    print(f"  Data device:    {data_device}")
    print(f"  Train device:   {train_device}")
    print(f"  Infer device:   {inference_device}")
    
    # 디렉토리 설정
    print(f"\n📁 Setting up directories...")
    dirs = setup_directories(base_dir)
    
    # Step 1: 데이터 준비
    if not skip_data_prep:
        cache_path = step1_prepare_data(
            gcm_hist_path=gcm_hist_path,
            obs_hist_path=obs_hist_path,
            dirs=dirs,
            n_quantiles=n_quantiles,
            lr_range=lr_range,
            hr_range=hr_range,
            device=data_device,
            force_recompute=force_recompute
        )
    else:
        print(f"\n⏭️  Skipping data preparation...")
        cache_filename = f'quantile_maps_q{n_quantiles}_{lr_range[0]}_{lr_range[1]}.pkl'
        cache_path = os.path.join(dirs['cache'], cache_filename)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache not found: {cache_path}")
        print(f"   Using existing cache: {cache_path}")
    
    # Step 2: 학습
    if not skip_training:
        config = get_default_config(use_fixed_range=True)
        config['num_epochs'] = num_epochs
        
        checkpoint_path = step2_train_model(
            cache_path=cache_path,
            dirs=dirs,
            config=config,
            device=train_device
        )
    else:
        print(f"\n⏭️  Skipping training...")
        checkpoint_path = os.path.join(dirs['checkpoints'], 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"   Using existing checkpoint: {checkpoint_path}")
    
    # Step 3: 추론
    results = step3_inference(
        ssp_paths=ssp_paths,
        checkpoint_path=checkpoint_path,
        cache_path=cache_path,
        dirs=dirs,
        device=inference_device,
        batch_size=32,
        var_name='tasmax'
    )
    
    # 최종 요약
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETED!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Data prepared:  {cache_path}")
    print(f"   Model trained:  {checkpoint_path}")
    print(f"   Scenarios processed: {len(results)}/{len(ssp_paths)}")
    
    if results:
        print(f"\n📂 Output files:")
        for scenario in results.keys():
            output_path = os.path.join(dirs['outputs'], f'vit_{scenario}_corrected.nc')
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / 1024**2
                print(f"   {scenario:10s}: {output_path} ({size_mb:.2f} MB)")
    
    print("\n" + "="*80 + "\n")
    
    return {
        'cache_path': cache_path,
        'checkpoint_path': checkpoint_path,
        'results': results,
        'dirs': dirs
    }


