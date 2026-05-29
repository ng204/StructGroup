import torch.optim


def build_optimizer(model, optim_cfg):
    assert 'type' in optim_cfg
    _optim_cfg = optim_cfg.copy()
    optim_type = _optim_cfg.pop('type')
    
    # 确保数值参数被正确转换为浮点数（处理YAML解析可能将科学计数法解析为字符串的情况）
    numeric_params = ['lr', 'weight_decay', 'eps', 'betas', 'momentum', 'alpha', 'rho', 'lambda']
    for param in numeric_params:
        if param in _optim_cfg:
            value = _optim_cfg[param]
            if isinstance(value, str):
                try:
                    _optim_cfg[param] = float(value)
                except ValueError:
                    pass  # 如果无法转换，保持原值（可能是其他格式，如betas是tuple）
            elif param == 'betas' and isinstance(value, (list, tuple)):
                # betas 是元组，需要转换每个元素
                _optim_cfg[param] = tuple(float(v) if isinstance(v, str) else v for v in value)
    
    optim = getattr(torch.optim, optim_type)
    return optim(filter(lambda p: p.requires_grad, model.parameters()), **_optim_cfg)
