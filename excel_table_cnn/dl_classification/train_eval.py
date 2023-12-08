from torch.utils.data import DataLoader
import torch

from .model.model import TableDetectionModel


def get_model(num_classes=2):
    model = TableDetectionModel(num_classes)
    return model


def train_model(model, train_loader, optimizer, num_epochs, device):
    # Send the model to the device (GPU or CPU)
    model.to(device)
    
    # Set the model in training mode
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets_list in train_loader:
            image = images[0]
            targets = targets_list[0]

            # Since the model expects a list of images, and list of targets, we repackage them
            images = [image.to(device)]  # Make it a list again to fit model input expectations
            targets = [{k: v.to(device) for k, v in targets.items()}]  # List of one dictionary
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()
            
            # Backpropagation
            losses.backward()
            optimizer.step()
        
        print(f"Epoch {epoch}: Loss: {epoch_loss / len(train_loader)}")

        
def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()  # Set the model in inference mode
    
    eval_loss = 0
    for images, targets_list in test_loader:
        image = images[0]
        targets = targets_list[0]
        images = [image.to(device)]
        targets = [{k: v.to(device) for k, v in targets.items()}]
        
        with torch.no_grad():
            # Forward pass
            loss_dict = model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
            eval_loss += losses.item()
    
    print(f"Test Loss: {eval_loss / len(test_loader)}")
