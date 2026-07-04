"""Run the chest X-ray pipeline: train (2 phases) + evaluate.

All settings live in src/config.py. Just run:  python main.py
"""
import torch

from src import config, data, pipeline


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch {torch.__version__} | device: {device}")

    train_loader, val_loader, test_loader, class_names = data.build_loaders()
    print("Classes:", class_names)

    weight = data.class_weights(class_names)
    print("Class weights:", {n: round(w, 3) for n, w in zip(class_names, weight.tolist())})

    model = pipeline.CXRClassifier(len(class_names)).to(device)
    history = pipeline.train(model, train_loader, val_loader, device, weight, len(class_names))
    pipeline.plot_history(history)

    if config.BEST_MODEL_PATH.exists():
        model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    pipeline.evaluate(model, test_loader, class_names, device)


if __name__ == "__main__":
    main()
