import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- batched slowness_loss ----------
def pairwise_loss(latent1: torch.Tensor,
                  latent2: torch.Tensor,
                  valid_mask: torch.Tensor,
                  pos_window: int = 5,
                  threshold: float = 0.0) -> torch.Tensor:
    """
    latent: [B, L, D]
    mask:   [B, L]     1 = valid, 0 = padding
    只对同 batch 且 |Δt| ≤ pos_window 的 pair 最小化 L2。
    阈值 threshold: 只计算距离大于阈值的部分的损失。
    """
    B, L, D = latent1.shape
    device = latent1.device

    lat1_flat  = latent1.reshape(-1, D)
    lat2_flat  = latent2.reshape(-1, D)
    mask_flat  = valid_mask.reshape(-1).bool()
    t_idx      = torch.arange(L, device=device).repeat(B)
    b_idx      = torch.arange(B, device=device).repeat_interleave(L)

    lat1_valid = lat1_flat[mask_flat]   # [N, D]
    lat2_valid = lat2_flat[mask_flat]   # [N, D]
    t_valid    = t_idx[mask_flat]       # [N]
    b_valid    = b_idx[mask_flat]       # [N]

    N = lat1_valid.size(0)
    if N < 2:
        return torch.tensor(0.0, device=device)

    dist = torch.cdist(lat1_valid, lat2_valid, p=2)  # [N, N]

    dt = torch.abs(t_valid.unsqueeze(1) - t_valid.unsqueeze(0))  # [N, N]
    same_batch = (b_valid.unsqueeze(1) == b_valid.unsqueeze(0))  # [N, N]
    valid_pair = same_batch & (dt <= pos_window)
    valid_pair &= ~torch.eye(N, dtype=torch.bool, device=device)

    if valid_pair.sum() == 0:
        return torch.tensor(0.0, device=device)

    # Linear decay: weight = 1 - dt / pos_window
    # decay_weight = 1.0 - dt.float() / (pos_window + 1)
    # decay_weight = decay_weight.clamp(min=0.0)  # avoid negative weights
    decay_weight = 1.0
    weighted_loss = dist * decay_weight * valid_pair.float()
    effective_weight = decay_weight * valid_pair.float()
    weighted_loss = F.relu(weighted_loss - threshold)
    return weighted_loss.sum() / effective_weight.sum()


# def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, 
#              mu: torch.Tensor, logvar: torch.Tensor, 
#              mask: torch.Tensor, beta: float = 1.0):
#     kl_per_timestep = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
#     # 对 latent 维度求和，形状变为 [B, L]
#     kl_per_timestep = kl_per_timestep.sum(dim=-1)
#     kl_per_timestep = kl_per_timestep * mask
#     valid_count = mask.sum()
#     loss_kl = kl_per_timestep.sum() / valid_count
#     return loss_kl

def uniform_loss(x: torch.Tensor, mask: torch.Tensor, t: float = 2) -> torch.Tensor:
    """
    计算均匀性损失，只使用 mask 中有效的 latent。
    
    参数:
        x: torch.Tensor, 形状为 [B, L, D]
            latent 表示。
        mask: torch.Tensor, 形状为 [B, L]
            padding mask，1 表示有效，0 表示无效。
        t: float, 默认值为 2，控制 exp 里的缩放因子。
    
    返回:
        torch.Tensor: 均匀性损失（标量）。
    """
    B, L, D = x.shape
    # reshape 为 [B*L, D]
    x_flat = x.reshape(-1, D)
    # 对应的 mask reshape 为 [B*L]
    mask_flat = mask.reshape(-1).bool()
    # 选择所有有效点
    x_valid = x_flat[mask_flat]
    
    # 如果有效点不足2个，则返回0损失
    if x_valid.shape[0] < 2:
        return torch.tensor(0.0, device=x.device)
    
    # 计算所有有效点之间的 pairwise distance
    dist = torch.pdist(x_valid, p=2)
    # 计算均匀性损失
    loss = dist.pow(2).mul(-t).exp().mean().log()
    return loss

def uniform_loss_temporal(x: torch.Tensor,
                          mask: torch.Tensor,
                          pos_window: int = 2,
                          neg_gap: int = 10,
                          t: float = 2.0,
                          dist_threshold: float = 100.0,
                          bank_samples: torch.Tensor = None,
                          max_points: int = None,
                          chunk_size: int = 512) -> torch.Tensor:
    """
    只用时间距离构造 repulsion，对 |Δt| >= neg_gap 的 pair 推开。
    当 latent 空间距离大于 dist_threshold 时不再产生 repulsion。
    
    Args:
        x:    [B, L, D] latent
        mask: [B, L]    1 有效 0 无效
        pos_window: 邻域大小，|Δt| <= pos_window 认为"近"，不推开
        neg_gap:    |Δt| >= neg_gap 认为"远"，强推开
        t:          缩放因子，越大 repulsion 越强
        dist_threshold: latent 空间距离阈值，超过此值不再推开
        bank_samples: [M, D] 可选，从 bank 中获取的历史样本，用于增加负样本
        max_points: optional deterministic subsample cap for valid latent points
        chunk_size: number of valid latent points per distance chunk
    Returns:
        scalar loss
    """
    B, L, D = x.shape
    device = x.device

    # flatten
    x_flat   = x.reshape(-1, D)                    # [B*L, D]
    mask_flat = mask.reshape(-1).bool()            # [B*L]
    time_idx = torch.arange(L, device=device).repeat(B)  # [B*L]
    batch_idx = torch.arange(B, device=device).repeat_interleave(L)

    # 仅保留有效帧
    x_valid   = x_flat[mask_flat]                  # [N, D]
    t_valid   = time_idx[mask_flat]                # [N]
    b_valid   = batch_idx[mask_flat]               # [N]

    N = x_valid.size(0)
    if N < 2:
        return torch.tensor(0.0, device=device)

    if max_points is not None and int(max_points) > 0 and N > int(max_points):
        sample_idx = torch.linspace(0, N - 1, int(max_points), device=device).long()
        x_valid = x_valid[sample_idx]
        t_valid = t_valid[sample_idx]
        b_valid = b_valid[sample_idx]
        N = x_valid.size(0)

    total_loss = torch.tensor(0.0, device=device)
    total_count = torch.tensor(0.0, device=device)
    all_idx = torch.arange(N, device=device)
    chunk_size = max(1, int(chunk_size))

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        x_chunk = x_valid[start:end]
        t_chunk = t_valid[start:end]
        b_chunk = b_valid[start:end]

        dist_x = torch.cdist(x_chunk, x_valid, p=2)
        dt = torch.abs(t_chunk.unsqueeze(1) - t_valid.unsqueeze(0))
        same_batch = b_chunk.unsqueeze(1) == b_valid.unsqueeze(0)
        weights = ((dt >= neg_gap) & same_batch) | (~same_batch)

        chunk_idx = torch.arange(start, end, device=device).unsqueeze(1)
        weights = weights & (chunk_idx != all_idx.unsqueeze(0))
        weights = weights & (dist_x < dist_threshold)

        if weights.any():
            weights_float = weights.float()
            total_loss = total_loss + (torch.exp(-t * dist_x.pow(2)) * weights_float).sum()
            total_count = total_count + weights_float.sum()
    
    # 如果有 bank samples，计算当前样本与 bank 样本之间的 repulsion
    if bank_samples is not None and bank_samples.size(0) > 0:
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            dist_to_bank = torch.cdist(x_valid[start:end], bank_samples, p=2)
            bank_mask = (dist_to_bank < dist_threshold).float()

            if bank_mask.sum() > 0:
                total_loss = total_loss + (torch.exp(-t * dist_to_bank.pow(2)) * bank_mask).sum()
                total_count = total_count + bank_mask.sum()
    
    if total_count == 0:
        return torch.tensor(0.0, device=device)
    
    loss = total_loss / total_count
    return loss.log()


def uniform_loss_pose_sensitive(x: torch.Tensor,
                                 pose: torch.Tensor,
                                 mask: torch.Tensor,
                                 t: float = 2.0,
                                 pose_threshold: float = 0.3) -> torch.Tensor:
    """
    改进的 uniform loss：只在 pose 距离足够远时推开 latent。
    
    参数:
        x: [B, L, D] latent 表示
        pose: [B, L, P] pose 表示（例如 x,y,θ）
        mask: [B, L] padding mask（1 表示有效）
        t: 缩放因子
        pose_threshold: pose 距离大于此值时才参与推开
    
    返回:
        torch.Tensor: 均匀性损失（标量）
    """
    B, L, D = x.shape
    x_flat = x.reshape(-1, D)           # [B*L, D]
    pose_flat = pose.reshape(-1, pose.shape[-1])  # [B*L, P]
    mask_flat = mask.reshape(-1).bool()           # [B*L]

    # 选择有效位置
    x_valid = x_flat[mask_flat]         # [N, D]
    pose_valid = pose_flat[mask_flat]   # [N, P]

    N = x_valid.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=x.device)

    # 构造 pairwise index
    dist_x = torch.cdist(x_valid, x_valid, p=2)     # [N, N]
    dist_pose =  torch.cdist(pose_valid[:, :2], pose_valid[:, :2], p=2)   # [N, N]

    # mask: 只保留 pose 距离大于 threshold 的 pair
    pose_mask = (dist_pose > pose_threshold).float()

    # 计算 loss（避免 diagonal，乘一个 diag_mask）
    diag_mask = 1.0 - torch.eye(N, device=x.device)
    weights = pose_mask * diag_mask

    loss_matrix = torch.exp(-t * dist_x.pow(2)) * weights
    loss = loss_matrix.sum() / weights.sum().clamp(min=1.0)

    return loss.log()


from scipy.spatial import procrustes
def calc_pas(latent: torch.Tensor, pose: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    计算 Procrustes Analysis 误差，使用 padding mask 过滤无效帧。
    
    参数:
        latent: torch.Tensor, 形状为 [B, L, 4]
            学到的 latent 表示。
        pose: torch.Tensor, 形状为 [B, L, 4]
            ground truth pose，格式为 [x, y, cos, sin]。
        mask: torch.Tensor, 形状为 [B, L]
            padding mask，1 表示有效帧，0 表示无效帧。
    
    返回:
        dict: 包含两个键：
            'trajectories': 每条 trajectory 分别计算对齐误差后取平均（只考虑有效帧）。
            'total': 将所有 trajectory 中的有效点展平后计算全局对齐误差。
    """
    latent_np = latent.cpu().detach().numpy()  # shape: [B, L, 4]
    pose_np = pose.cpu().detach().numpy()      # shape: [B, L, 4]
    mask_np = mask.cpu().detach().numpy()      # shape: [B, L]
    
    # print(f"latent_np shape: {latent_np.shape}, pose_np shape: {pose_np.shape}, mask_np shape: {mask_np.shape}")
    errors = []
    B = latent_np.shape[0]
    
    # 对每条 trajectory 单独计算 Procrustes Analysis 的残差，仅考虑有效的时间步
    for i in range(B):
        valid = mask_np[i].astype(bool)  # [L] 内部标识有效帧
        if np.sum(valid) < 2:
            # 有效帧不足，跳过该 trajectory
            continue
        latent_valid = latent_np[i][valid]
        pose_valid = pose_np[i][valid]
        # 使用 scipy.spatial.procrustes 计算对齐
        _, _, disparity = procrustes(pose_valid, latent_valid)
        errors.append(disparity)
    
    # 如果至少有一条 trajectory 有足够的有效帧，则计算平均误差；否则返回 NaN
    traj_error = np.mean(errors) if len(errors) > 0 else np.nan

    # 全局误差：将所有 trajectory 中有效的时间步展平成一个 [N, 4] 的数组，再计算 Procrustes Analysis
    valid_global = mask_np.astype(bool)  # shape: [B, L]
    latent_flat = latent_np[valid_global]  # 结果 shape: [N, 4]
    pose_flat = pose_np[valid_global]
    
    if latent_flat.shape[0] < 2:
        global_error = np.nan  # 有效点不足
    else:
        _, _, global_error = procrustes(pose_flat, latent_flat)
    
    return {"trajectories": traj_error, "total": global_error}
