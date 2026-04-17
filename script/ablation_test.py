"""Run quick sanity tests for ablation MeanMax models.

Creates each model, prints parameter counts and verifies a forward pass
on a dummy tensor.
"""
import torch
try:
    from src.model.ablation import (
        create_cnn_meanmax_model,
        create_tcn_meanmax_model,
        create_vit_meanmax_model,
        create_inceptiontime_meanmax_model,
        create_lstm_meanmax_model,
        create_cnn_lstm_meanmax_model,
        create_cnn_transformer_meanmax_model,
        create_mamba_meanmax_model,
    )
except Exception:
    # Fallback: load ablation module directly by path to avoid importing top-level `src` package
    import importlib.util
    import types
    import sys
    import os
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    src_dir = os.path.join(base, 'src')
    # Create lightweight package entries to support relative imports in the module
    src_pkg = types.ModuleType('src')
    src_pkg.__path__ = [src_dir]
    sys.modules['src'] = src_pkg
    model_pkg = types.ModuleType('src.model')
    model_pkg.__path__ = [os.path.join(src_dir, 'model')]
    sys.modules['src.model'] = model_pkg

    module_path = os.path.join(src_dir, 'model', 'ablation', 'meanmax_models.py')
    spec = importlib.util.spec_from_file_location('src.model.ablation.meanmax_models', module_path)
    ablation_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ablation_mod)
    create_cnn_meanmax_model = ablation_mod.create_cnn_meanmax_model
    create_tcn_meanmax_model = ablation_mod.create_tcn_meanmax_model
    create_vit_meanmax_model = ablation_mod.create_vit_meanmax_model
    create_inceptiontime_meanmax_model = ablation_mod.create_inceptiontime_meanmax_model
    create_lstm_meanmax_model = ablation_mod.create_lstm_meanmax_model
    create_cnn_lstm_meanmax_model = ablation_mod.create_cnn_lstm_meanmax_model
    create_cnn_transformer_meanmax_model = ablation_mod.create_cnn_transformer_meanmax_model
    create_mamba_meanmax_model = ablation_mod.create_mamba_meanmax_model


def model_info(model):
    params = sum(p.numel() for p in model.parameters())
    return params


def run_single(create_fn, name: str, seq_len: int = 512, device: str = 'cpu'):
    try:
        model = create_fn(input_length=seq_len, num_classes=10, device=device)
        model.eval()
        params = model_info(model)
        print(f"Model: {name} | Params: {params}")
        x = torch.randn(2, 1, seq_len).to(device)
        with torch.no_grad():
            out = model(x)
        print(f"  Forward output shape: {tuple(out.shape)}")
    except Exception as e:
        print(f"  ERROR running {name}: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seq-len', default=512, type=int)
    args = parser.parse_args()

    tests = [
        (create_cnn_meanmax_model, 'SpectrumCNNMeanMax'),
        (create_tcn_meanmax_model, 'SpectrumTCNMeanMax'),
        (create_vit_meanmax_model, 'SpectrumViTMeanMax'),
        (create_inceptiontime_meanmax_model, 'SpectrumInceptionTimeMeanMax'),
        (create_lstm_meanmax_model, 'SpectrumLSTMMeanMax'),
        (create_cnn_lstm_meanmax_model, 'SpectrumCNNLSTMMeanMax'),
        (create_cnn_transformer_meanmax_model, 'SpectrumCNNTransformerMeanMax'),
        (create_mamba_meanmax_model, 'SpectrumMambaS4MeanMax'),
    ]

    for fn, name in tests:
        run_single(fn, name, seq_len=args.seq_len, device=args.device)


if __name__ == '__main__':
    main()
