import torch
import torch.nn as nn
import numpy as np

class WeatherTransformer(nn.Module):
    def __init__(self, num_features, num_heads, num_layers, dropout=0.1):
        super(WeatherTransformer, self).__init__()
        # self.model_type = 'Transformer'
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=num_features, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(num_features, 1)
    
    def forward(self, src):
        # src shape: (batch_size, sequence_length, num_features)
        src = src.permute(1, 0, 2)  # transformer expects (sequence_length, batch_size, num_features)
        output = self.transformer_encoder(src)
        output = self.decoder(output[-1])  # Use the output of the last time step
        return output

# Ustawienia modelu
num_features = 10  # Liczba cech w danych wejściowych
num_heads = 2  # Liczba głów uwagi
num_layers = 12  # Liczba warstw enkodera
dropout = 0.1  # Współczynnik dropoutu

model = WeatherTransformer(num_features, num_heads, num_layers, dropout)
print(model)

from torch.utils.data import DataLoader, TensorDataset

# Przykład danych (dummy data)
X_train = np.random.rand(1000, 24, num_features).astype(np.float32)  # 1000 próbek, sekwencje 24-godzinne, 10 cech
y_train = np.random.rand(1000, 1).astype(np.float32)  # 1000 etykiet

# Konwersja do tensora
train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
