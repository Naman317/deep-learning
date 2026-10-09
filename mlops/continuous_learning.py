import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def get_weight_path(ticker, model_type="lstm"):
    clean_sym = ticker.replace(".NS", "_NS").replace(".BO", "_BO")
    return os.path.join(WEIGHTS_DIR, f"{clean_sym}_{model_type.lower()}.pth")


def load_saved_weights(model, ticker, model_type="lstm"):
    path = get_weight_path(ticker, model_type)
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            return model, True
        except Exception:
            return model, False
    return model, False


def save_model_weights(model, ticker, model_type="lstm"):
    path = get_weight_path(ticker, model_type)
    try:
        torch.save(model.state_dict(), path)
        return True
    except Exception:
        return False
