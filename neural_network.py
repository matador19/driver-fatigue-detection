import torchvision
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
# Source: https://www.cs.toronto.edu/~kriz/cifar.html
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

TRAIN_DIR = os.path.join(os.path.dirname(__file__), "dataset")
BATCH_SIZE = 32
TESTING_DATASET = os.path.join(os.path.dirname(__file__), "dataset")


# Define transformations for training and testing dataset
# Look at available transformations at https://pytorch.org/vision/stable/transforms.html
# Training transformation can modify input data in any (sensible) way
training_transformation = transforms.Compose(
    [ ### FILL ###
     transforms.ToTensor()])

# Testing transformation should only normalize the input data - DO NOT CHANGE THIS LINE!
testing_transformation = transforms.Compose([transforms.ToTensor()])


def show_batch(batch_data):
    # Prepare grid of images and create numpy array from them
    image = torchvision.utils.make_grid(batch_data[:16], nrow=4)
    image = image.numpy().transpose(1, 2, 0)
    plt.grid(False)

    # Display image
    plt.imshow(image)

# Create DataLoader and obtain the first batch of the dataset


data = datasets.ImageFolder(TRAIN_DIR, transform=training_transformation)
train_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

batch_x, batch_y = next(iter(train_loader))
print(np.shape(batch_x), batch_y)

#  batch_data, batch_labels = next(iter(train_loader))

# Print useful information about the batch and show the data
print("Shapes")
print(f"- data: {batch_x.shape}")
print(f"- labels: {batch_y}")
show_batch(batch_x)

# def draw_progress(data):
#   iterations = [item[0] for item in data]
#   training_loss = [item[1] for item in data]
#   training_accuracy = [item[2] * 100 for item in data]
#   testing_loss = [item[3] for item in data]
#   testing_accuracy = [item[4] * 100 for item in data]
#
#   plt.plot(iterations, training_loss, label='Training')
#   plt.plot(iterations, testing_loss, label='Testing')
#   plt.legend()
#   plt.ylim([0, None])
#   plt.title('Loss')
#   plt.show()
#
#   plt.plot(iterations, training_accuracy, label='Training')
#   plt.plot(iterations, testing_accuracy, label='Testing')
#   plt.legend()
#   plt.ylim([0, 100])
#   plt.title('Accuracy')
#   plt.show()
#
# def print_progress(data):
#   for (iteration, training_loss, training_accuracy, testing_loss, testing_accuracy) in data:
#     print(f"Iteration:{iteration} Loss:{training_loss:.3f}|{testing_loss:.3f} Accuracy:{100*training_accuracy:.2f}%|{100*testing_accuracy:.2f}%")
#
#     def test(model, data_loader):
#         model = model.eval()
#
#         # Accumulators
#         loss_acc = 0
#         accuracy_acc = 0
#         counter = 0
#
#         # Loop through dataset
#         for batch_data, batch_labels in data_loader:
#             # Calculate output
#             logits = model(batch_data)
#
#             # Accumulate loss value
#             loss_acc += criterion(logits, batch_labels).item()
#
#             # Compute and accumulate ratio of correct label predictions
#             max_scores, pred_labels = torch.max(logits, dim=1)
#             accuracy_acc += torch.sum(pred_labels == batch_labels).item() / batch_data.shape[0]
#             counter += 1
#
#         model = model.train()
#         return loss_acc / counter, accuracy_acc / counter
#
#     def training_step(model, input_data, target_labels, criterion, optimizer):
#         input_data = input_data.to('cuda')
#         target_labels = target_labels.to('cuda')
#
#         # Forward pass - compute network autput and store all activations
#         outputs = model(input_data)
#
#         # Compute loss
#         loss = criterion(outputs, target_labels)
#
#         # Backward pass - compute gradients
#         optimizer.zero_grad()
#         loss.backward()
#
#         # Optimize network
#         optimizer.step()
#
#         # .item() and .detach() disconects from comp. graph
#         return loss.cpu().item(), outputs.detach().cpu()
#
#     def create_network():
#         network = nn.Sequential(
#             #
#             # Input: (N, 3, 32, 32)
#             #
#             nn.Conv2d(3, 12, (3, 3), padding='same'),
#             nn.ReLU(),
#             nn.Conv2d(12, 8, (3, 3), padding='same'),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             #
#             # (N, 8, 16, 16)
#             #
#             #
#             nn.Conv2d(8, 16, (3, 3), padding='same'),
#             nn.ReLU(),
#             nn.Conv2d(16, 16, (3, 3), padding='same'),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             # (N, 16, 8, 8)
#             #
#             #
#             nn.Conv2d(16, 32, (3, 3), padding='same'),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, (3, 3), padding='same'),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             # (N, 32, 4, 4)
#             #
#             nn.Flatten(),
#             #
#             # (N, 16*32)
#             #
#             #
#             nn.Linear(in_features=16 * 32, out_features=10)
#
#             # (N, 10)
#             #
#         )
#
#         return network
#
#     # Create new model and print its structure
#     model = create_network()
#     print(model)
#
#     # Training setup
#     batch_size = 16
#     view_step = 500
#     iterations = 25000
#
#     # Create training and testing DataLoaders
#     training_loader = torch.utils.data.DataLoader(TRAIN_DIR, batch_size=batch_size, shuffle=True,
#                                                   pin_memory=True, num_workers=2)
#     testing_loader = torch.utils.data.DataLoader(TESTING_DATASET, batch_size=4, shuffle=True, pin_memory=True,
#                                                  num_workers=2)
#
#     # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss().to('cuda')
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     model = model.to('cuda')
#
#     # Accumulators
#     loss_acc = 0
#     accuracy_acc = 0
#
#     training_progress = []
#
#     iteration = 0
#     stop_training = False
#
#     # When we reach the end of the dataset, but do not want to end the training, we loop through it again
#     while not stop_training:
#         # Obtain batch from the training dataset in a loop
#         for batch_data, batch_labels in training_loader:
#             iteration += 1
#
#             # Do a training step
#             loss, outputs = training_step(model, batch_data, batch_labels, criterion, optimizer)
#
#             # Accumulate loss for statistics
#             loss_acc += loss
#
#             # Compute and accumulate ratio of correct label predictions for statistics
#             max_scores, pred_labels = torch.max(outputs, 1)
#             accuracy_acc += torch.sum(pred_labels == batch_labels).item() / float(batch_size)
#
#             # Test model
#             if iteration % view_step == 0:
#                 model = model.to('cpu')
#                 # Calculate loss and accuracy on the testing dataset
#                 test_loss_acc, test_accuracy_acc = test(model, testing_loader)
#                 training_progress.append(
#                     (iteration, loss_acc / view_step, accuracy_acc / view_step, test_loss_acc, test_accuracy_acc))
#
#                 # Clear output window and show training progress
#                 clear_output()
#                 draw_progress(training_progress)
#                 print_progress(training_progress)
#
#                 # Reset the accumulators
#                 loss_acc = 0
#                 accuracy_acc = 0
#                 model = model.to('cuda')
#
#             # Stop training when amount of iterations is reached
#             if iteration >= iterations:
#                 stop_training = True
#                 break
#
#     print("Training finished.")
#
#     def print_output(data):
#         # Calculate probabilities from logits
#         data = nn.functional.softmax(data, dim=1)
#         # Calculate maxima and labels
#         max_scores, pred_labels = torch.max(data, dim=1)
#
#         # Print output line by line
#         for (label, score, probs) in zip(pred_labels, max_scores, data):
#             probs = f"[ {', '.join([f'{prob:.2f}' for prob in probs])} ]"
#             print(f"{probs} ... {label} ({classes[label]}) [{score * 100:.2f}%]")
#
#     model = model.eval()
#
#     # Get data from tesing dataset
#     batch_data, batch_labels = iter(testing_loader).next()
#     # Show data
#     show_batch(batch_data)
#     # Run model and print output
#     print_output(model(batch_data).detach())
#
#     model = model.train()