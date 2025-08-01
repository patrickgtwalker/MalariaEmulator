import torch 
import torch.nn as nn
import torch.optim as optim
import time


#Function to define the ML model
class LSTMModel(nn.Module):
    def __init__(self, input_size, architecture):
        super(LSTMModel, self).__init__()
        self.lstm_layers = nn.ModuleList()
        for i, hidden_size in enumerate(architecture):
            self.lstm_layers.append(nn.LSTM(input_size if i == 0 else architecture[i - 1], hidden_size, batch_first=True))
        self.fc = nn.Linear(architecture[-1], 2)
    
    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = self.fc(x[:, -1, :])
        return x
    

#Function to train the model

def train_model(model, train_loader, eval_loader, model_name, epochs=25, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []
    eval_loss_history = []

    start_time = time.time()  # Track time

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch.unsqueeze(-1))
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Evaluation phase
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for X_val, y_val in eval_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val.unsqueeze(-1))
                eval_loss += criterion(val_outputs, y_val).item()

        avg_eval_loss = eval_loss / len(eval_loader)
        eval_loss_history.append(avg_eval_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

    duration = time.time() - start_time  # Training duration

    # Save the trained model
    torch.save(model.state_dict(), f"src/{model_name}_model.pth")
    print(f"Model saved as src/{model_name}_model.pth")

    return model, loss_history, eval_loss_history, duration