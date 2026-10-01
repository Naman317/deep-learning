import torch
import torch.nn as nn
import numpy as np


def train_model(model, train_loader, epochs=15, optimizer_type="Adam", lr=0.001, patience=5, callback=None):
    criterion = nn.MSELoss()
    
    if optimizer_type == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr * 10, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    epoch_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        
        if callback:
            callback(epoch + 1, epochs, avg_loss)
            
    return model, epoch_losses


def autoregressive_forecast(model, last_sequence, scaler, seq_length=60, num_features=3, forecast_horizon=7):
    # Iteratively forecast next steps using previous output predictions
    model.eval()
    future_predictions = []
    current_seq = last_sequence.copy()
    
    with torch.no_grad():
        for _ in range(forecast_horizon):
            X_test = np.reshape(current_seq, (1, seq_length, num_features))
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            pred_scaled = model(X_test_tensor).numpy()[0][0]
            
            dummy = np.zeros((1, num_features))
            dummy[0, 0] = pred_scaled
            pred_unscaled = scaler.inverse_transform(dummy)[0][0]
            future_predictions.append(float(pred_unscaled))
            
            new_row = current_seq[-1].copy()
            new_row[0] = pred_scaled
            current_seq = np.vstack([current_seq[1:], new_row])
            
    return future_predictions
