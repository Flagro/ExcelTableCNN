import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from cnn import FCN_RCNN

def evaluate_cnn(train_dataset, test_dataset, num_features, num_epochs=10):
    # Initialize the model
    model = FCN_RCNN(num_features=num_features).to('cuda')  # Move model to GPU

    # Define the loss function and optimizer
    # You might need to adjust the loss function based on the exact output of your model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):  # num_epochs is the number of epochs
        model.train()
        for images, targets in train_loader:
            images = images.to('cuda')  # Move images to GPU
            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]  # Move targets to GPU

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Perform backpropagation
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to('cuda')
            targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
            predictions = model(images)

    # Save the trained model
    torch.save(model.state_dict(), '/mnt/data/trained_model.pth')

    # To load the model later, use:
    # model.load_state_dict(torch.load('/mnt/data/trained_model.pth'))
