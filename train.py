# train.py
import torch
from torch.utils.data import DataLoader
from model import SimpleViT
from engine import BiasCorrectTrainer, QuantileDataset


def train_bias_correction(
    lr_q_train, hr_q_train, 
    lr_q_val, hr_q_val,
    quantiles, 
    config,
    normalization=None,  # ⭐ 새 파라미터
    lr_stats=None,  # Deprecated (하위 호환성)
    hr_stats=None   # Deprecated (하위 호환성)
):
    """
    편이보정 학습
    
    Parameters:
    -----------
    lr_q_train, hr_q_train : numpy.ndarray
        훈련 분위수 맵 [n_quantiles, H, W], [0, 1] 정규화됨
    lr_q_val, hr_q_val : numpy.ndarray
        검증 분위수 맵 [n_quantiles, H, W], [0, 1] 정규화됨
    quantiles : numpy.ndarray
        분위수 배열 [0, 100]
    config : dict
        학습 설정
    normalization : dict
        정규화 정보 {'type': 'fixed_range', 'lr_min': 260, 'lr_max': 320, ...}
    """
    
    print("="*60)
    print("🚀 Starting Training Pipeline")
    print("="*60)
    
    # 1. 데이터 범위 확인
    print(f"\n📊 Input data info:")
    print(f"   LR train: {lr_q_train.shape}, range=[{lr_q_train.min():.3f}, {lr_q_train.max():.3f}]")
    print(f"   HR train: {hr_q_train.shape}, range=[{hr_q_train.min():.3f}, {hr_q_train.max():.3f}]")
    print(f"   LR val:   {lr_q_val.shape}, range=[{lr_q_val.min():.3f}, {lr_q_val.max():.3f}]")
    print(f"   HR val:   {hr_q_val.shape}, range=[{hr_q_val.min():.3f}, {hr_q_val.max():.3f}]")
    
    if normalization is not None:
        print(f"\n📊 Normalization:")
        print(f"   Type: {normalization['type']}")
        if normalization['type'] == 'fixed_range':
            print(f"   LR: [{normalization['lr_min']}K, {normalization['lr_max']}K]")
            print(f"   HR: [{normalization['hr_min']}K, {normalization['hr_max']}K]")
    
    # 2. Dataset 생성 (범위 검증 포함)
    print(f"\n📦 Creating datasets...")
    train_dataset = QuantileDataset(
        lr_q_train, hr_q_train, 
        verify_range=True  # ⭐ 범위 검증
    )
    val_dataset = QuantileDataset(
        lr_q_val, hr_q_val, 
        verify_range=True  # ⭐ 범위 검증
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=config['num_workers'], 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=config['num_workers'], 
        pin_memory=True
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    
    # 3. 모델 생성
    print(f"\n🔧 Creating model...")
    
    # ⭐ Sigmoid 사용 여부 결정
    use_sigmoid = (normalization is not None and 
                   normalization.get('type') == 'fixed_range')
    
    model = SimpleViT(
        img_size=lr_q_train.shape[1],
        patch_size=config['patch_size'],
        in_chans=1,
        out_chans=1,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=4.,
        dropout=config['dropout'],
        use_sigmoid=use_sigmoid  # ⭐ 고정 범위면 True
    )
    
    print(f"   Model: {type(model).__name__}")
    print(f"   Image size: {lr_q_train.shape[1]}x{lr_q_train.shape[2]}")
    print(f"   Patch size: {config['patch_size']}")
    print(f"   Embed dim: {config['embed_dim']}")
    print(f"   Depth: {config['depth']}")
    print(f"   Num heads: {config['num_heads']}")
    print(f"   Dropout: {config['dropout']}")
    print(f"   Use sigmoid: {use_sigmoid}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 4. Trainer 생성
    print(f"\n🎯 Creating trainer...")
    trainer = BiasCorrectTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config['device'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        checkpoint_dir=config['checkpoint_dir'],
        plot_dir=config['plot_dir'],
        plot_every=config['plot_every'],
        quantiles=quantiles,
        normalization=normalization,  # ⭐ 고정 범위 정보
        # 하위 호환성
        lr_stats=lr_stats,
        hr_stats=hr_stats
    )
    
    # 5. 학습률 스케줄러 설정
    trainer.set_scheduler('cosine', config['num_epochs'])
    
    # 6. 학습 시작
    print("\n" + "="*60)
    trainer.fit(config['num_epochs'], lr_q_val)
    
    return trainer


def get_default_config(use_fixed_range=True):
    """
    기본 설정
    
    Parameters:
    -----------
    use_fixed_range : bool
        고정 범위 방식 사용 여부
    """
    
    if use_fixed_range:
        # 고정 범위 방식 (더 안정적)
        return {
            'batch_size': 8,  # 약간 증가
            'num_workers': 4,
            
            # SimpleViT with Sigmoid
            'patch_size': 8,      # 적절한 크기
            'embed_dim': 512,     # 충분한 표현력
            'depth': 12,          # 깊이
            'num_heads': 8,       # 헤드 수
            'dropout': 0.1,       # 적당한 드롭아웃
            
            'num_epochs': 200,
            'lr': 1e-4,           # 적당한 학습률
            'weight_decay': 0.01,
            
            'device': 'cuda:1',
            'checkpoint_dir': 'checkpoints/vit_fixed_range',
            'plot_dir': 'plots/vit_fixed_range',
            'plot_every': 5
        }
    else:
        # 기존 방식 (Z-score)
        return {
            'batch_size': 4,
            'num_workers': 4,
            
            # SimpleViT without Sigmoid
            'patch_size': 4,
            'embed_dim': 256,
            'depth': 4,
            'num_heads': 4,
            'dropout': 0.1,
            
            'num_epochs': 200,
            'lr': 5e-5,
            'weight_decay': 0.01,
            
            'device': 'cuda:1',
            'checkpoint_dir': 'checkpoints/simple_vit',
            'plot_dir': 'plots/simple_vit',
            'plot_every': 5
        }


# ⭐ 편의 함수: 고정 범위 학습
def train_fixed_range(
    lr_q_train, hr_q_train,
    lr_q_val, hr_q_val,
    quantiles,
    lr_range=(260, 320),
    hr_range=(260, 320),
    custom_config=None
):
    """
    고정 범위 방식으로 학습
    
    Parameters:
    -----------
    lr_q_train, hr_q_train : numpy.ndarray
        훈련 분위수 맵 [n_quantiles, H, W], [0, 1] 정규화
    lr_q_val, hr_q_val : numpy.ndarray
        검증 분위수 맵 [n_quantiles, H, W], [0, 1] 정규화
    quantiles : numpy.ndarray
        분위수 배열
    lr_range, hr_range : tuple
        물리 단위 범위 (min, max) in Kelvin
    custom_config : dict or None
        사용자 정의 설정 (None이면 기본값 사용)
    """
    
    # 정규화 정보
    normalization = {
        'type': 'fixed_range',
        'lr_min': lr_range[0],
        'lr_max': lr_range[1],
        'hr_min': hr_range[0],
        'hr_max': hr_range[1]
    }
    
    # 설정
    if custom_config is None:
        config = get_default_config(use_fixed_range=True)
    else:
        config = custom_config
    
    # 학습
    trainer = train_bias_correction(
        lr_q_train, hr_q_train,
        lr_q_val, hr_q_val,
        quantiles=quantiles,
        config=config,
        normalization=normalization
    )
    
    return trainer


# ⭐ 편의 함수: 표준 방식 학습 (하위 호환성)
def train_standard(
    lr_q_train, hr_q_train,
    lr_q_val, hr_q_val,
    lr_stats, hr_stats,
    quantiles,
    custom_config=None
):
    """
    표준 방식으로 학습 (Z-score 정규화)
    
    하위 호환성을 위한 함수
    """
    
    # 설정
    if custom_config is None:
        config = get_default_config(use_fixed_range=False)
    else:
        config = custom_config
    
    # 학습
    trainer = train_bias_correction(
        lr_q_train, hr_q_train,
        lr_q_val, hr_q_val,
        quantiles=quantiles,
        config=config,
        normalization=None,  # 표준 방식
        lr_stats=lr_stats,
        hr_stats=hr_stats
    )
    
    return trainer


if __name__ == "__main__":
    # 사용 예시
    import pickle
    
    print("="*60)
    print("Training Script - Usage Example")
    print("="*60)
    
    # 1. 데이터 로드
    print("\n📂 Loading data...")
    with open('quantile_maps_fixed_range.pkl', 'rb') as f:
        qdata = pickle.load(f)
    
    lr_q_train, hr_q_train = qdata['train']
    lr_q_val, hr_q_val = qdata['val']
    quantiles = qdata['quantiles']
    normalization = qdata['normalization']
    
    print(f"   Loaded: {lr_q_train.shape}")
    
    # 2. 고정 범위 방식 학습
    print("\n🚀 Method 1: Fixed Range Training")
    
    trainer = train_fixed_range(
        lr_q_train, hr_q_train,
        lr_q_val, hr_q_val,
        quantiles=quantiles,
        lr_range=(normalization['lr_min'], normalization['lr_max']),
        hr_range=(normalization['hr_min'], normalization['hr_max']),
        custom_config=None  # 기본 설정 사용
    )
    
    print(f"\n✅ Training completed!")
    print(f"   Best loss: {trainer.best_loss:.6f}")
    
    # 3. 또는 직접 호출
    print("\n🚀 Method 2: Direct Call")
    
    config = get_default_config(use_fixed_range=True)
    config['num_epochs'] = 100  # 에폭 수정
    
    trainer2 = train_bias_correction(
        lr_q_train, hr_q_train,
        lr_q_val, hr_q_val,
        quantiles=quantiles,
        config=config,
        normalization=normalization
    )
    
    print("\n" + "="*60)