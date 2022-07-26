from PIL import Image
import torchvision
from torchvision import transforms
from torchvision import datasets
from torchvision.models import vgg16
import matplotlib.pyplot as plt
import os
import io
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from console import Console

CLASSES = ['eyes_closed', 'eyes_down', 'eyes_forward', 'eyes_left', 'eyes_right', 'eyes_up']

ROOT_DIR = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(ROOT_DIR, "dataset")
TEST_DIR = os.path.join(ROOT_DIR, "test")
MODEL_PATH = os.path.join(ROOT_DIR, 'model.pt')

BATCH_SIZE = 32

# Define transformations for training and testing dataset
# Look at available transformations at https://pytorch.org/vision/stable/transforms.html
# Training transformation can modify input data in any (sensible) way
training_transformation = transforms.Compose(
    [  ### FILL ###
        transforms.ToTensor()])
# Testing transformation should only normalize the input data - DO NOT CHANGE THIS LINE!
testing_transformation = transforms.Compose([transforms.ToTensor()])


def train():
    # Create new model and print its structure
    model = create_network()
    print(model)

    # Training setup
    batch_size = 16
    view_step = 500
    iterations = 10000

    # Create training and testing DataLoaders
    train_data = datasets.ImageFolder(TRAIN_DIR, transform=training_transformation)
    test_data = datasets.ImageFolder(TEST_DIR, transform=training_transformation)

    # Create DataLoader and obtain the first batch of the dataset
    training_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    testing_loader = DataLoader(test_data, batch_size=4, shuffle=True, pin_memory=True, num_workers=2)

    batch_data, batch_labels = next(iter(training_loader))
    # print(np.shape(batch_data), batch_labels)

    # Print useful information about the batch and show the data
    print("Shapes")
    print(f"- data: {batch_data.shape}")
    print(f"- labels: {batch_labels}")
    show_batch(batch_data)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = model.to('cuda')

    # Accumulators
    loss_acc = 0
    accuracy_acc = 0

    training_progress = []

    iteration = 0
    stop_training = False

    Console.info(logger="NN training", msg="Starting training")

    # When we reach the end of the dataset, but do not want to end the training, we loop through it again
    while not stop_training:
        # Obtain batch from the training dataset in a loop
        for batch_data, batch_labels in training_loader:
            print('Training iteration:', iteration, '/', iterations)
            iteration += 1

            # Do a training step
            loss, outputs = training_step(model, batch_data, batch_labels, criterion, optimizer)

            # Accumulate loss for statistics
            loss_acc += loss

            # Compute and accumulate ratio of correct label predictions for statistics
            max_scores, pred_labels = torch.max(outputs, 1)
            accuracy_acc += torch.sum(pred_labels == batch_labels).item() / float(batch_size)

            # Test model
            if iteration % view_step == 0:
                model = model.to('cpu')
                # Calculate loss and accuracy on the testing dataset
                test_loss_acc, test_accuracy_acc = test(model, criterion, testing_loader)
                training_progress.append((iteration, loss_acc / view_step, accuracy_acc / view_step, test_loss_acc, test_accuracy_acc))

                # Clear output window and show training progress
                #  clear_output()
                draw_progress(training_progress)
                print_progress(training_progress)

                # Reset the accumulators
                loss_acc = 0
                accuracy_acc = 0
                model = model.to('cuda')

            # Stop training when amount of iterations is reached
            if iteration >= iterations:
                stop_training = True
                break

    Console.info(logger="NN training", msg="Training finished.")
    Console.info(logger="NN training", msg=f"Saving trained model to: {MODEL_PATH}")
    torch.save(model.state_dict(), MODEL_PATH)
    Console.info(logger="NN training", msg="Model saved")


def model_eval(img):
    model = create_network()
    # model = model.to('cuda')

    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

    model = model.eval()

    # image normalization taken from:
    # https://discuss.pytorch.org/t/how-to-test-single-image-for-a-model-trained-with-dataloader-loaded-dataset/91246
    image = Image.fromarray(img)  # Webcam frames are numpy array format
    # Therefore transform back to PIL image
    image = training_transformation(image)
    image = image.float()
    # image = Variable(image, requires_autograd=True)
    # image = image.cuda()
    image = image.unsqueeze(0)  # I don't know for sure but Resnet-50 model seems to only
    # end

    output = model(image)
    # probs = torch.nn.functional.softmax(output, dim=1)
    conf, predicted = torch.max(output.data, 1)

    print(conf, CLASSES[predicted.item()])
    return CLASSES[predicted.item()]
    #  return conf.item(), index_to_breed[classes.item()]


def show_batch(batch_data):
    # Prepare grid of images and create numpy array from them
    image = torchvision.utils.make_grid(batch_data[:16], nrow=4)
    image = image.numpy().transpose(1, 2, 0)
    plt.grid(False)

    # Display image
    plt.imshow(image)


def draw_progress(data):
    iterations = [item[0] for item in data]
    training_loss = [item[1] for item in data]
    training_accuracy = [item[2] * 100 for item in data]
    testing_loss = [item[3] for item in data]
    testing_accuracy = [item[4] * 100 for item in data]

    plt.plot(iterations, training_loss, label='Training')
    plt.plot(iterations, testing_loss, label='Testing')
    plt.legend()
    plt.ylim([0, None])
    plt.title('Loss')
    plt.show()

    plt.plot(iterations, training_accuracy, label='Training')
    plt.plot(iterations, testing_accuracy, label='Testing')
    plt.legend()
    plt.ylim([0, 100])
    plt.title('Accuracy')
    plt.show()


def print_progress(data):
    for (iteration, training_loss, training_accuracy, testing_loss, testing_accuracy) in data:
        print(f"Iteration:{iteration} Loss:{training_loss:.3f}|{testing_loss:.3f} Accuracy:{100*training_accuracy:.2f}%|{100*testing_accuracy:.2f}%")


def test(model, criterion, data_loader):
    model = model.eval()

    # Accumulators
    loss_acc = 0
    accuracy_acc = 0
    counter = 0

    # Loop through dataset
    for batch_data, batch_labels in data_loader:
        # Calculate output
        logits = model(batch_data)

        # Accumulate loss value
        loss_acc += criterion(logits, batch_labels).item()

        # Compute and accumulate ratio of correct label predictions
        max_scores, pred_labels = torch.max(logits, dim=1)
        accuracy_acc += torch.sum(pred_labels == batch_labels).item() / batch_data.shape[0]
        counter += 1

    model = model.train()
    return loss_acc / counter, accuracy_acc / counter


def training_step(model, input_data, target_labels, criterion, optimizer):
    input_data = input_data.to('cuda')
    target_labels = target_labels.to('cuda')

    # Forward pass - compute network autput and store all activations
    outputs = model(input_data)

    # Compute loss
    loss = criterion(outputs, target_labels)

    # Backward pass - compute gradients
    optimizer.zero_grad()
    loss.backward()

    # Optimize network
    optimizer.step()

    # .item() and .detach() disconects from comp. graph
    return loss.cpu().item(), outputs.detach().cpu()


def create_network():
    number_of_layers = 17
    network = nn.Sequential(
        vgg16(pretrained=True).features[:number_of_layers],
        # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(in_features=512*4*8, out_features=len(CLASSES), bias=True)
        # nn.Flatten(start_dim=1, end_dim=2),
        # nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=3, bias=False),
        # nn.Linear(in_features=, out_features=len(CLASSES))
    )

    # print("Extracted layers")
    # print(network)
    # network = nn.Sequential(
    #     #
    #     # Input: (N, 3, 32, 64)
    #     #
    #     nn.Conv2d(3, 12, (3, 3), padding='same'),
    #     nn.ReLU(),
    #     nn.Conv2d(12, 8, (3, 3), padding='same'),
    #     nn.ReLU(),
    #     nn.MaxPool2d((2, 2)),
    #     # nn.Linear(in_features=, out_features=len(CLASSES))
    # )
    # network = nn.Sequential(
    #     #
    #     # Input: (N, 3, 32, 64)
    #     #
    #     nn.Conv2d(3, 12, (3, 3), padding='same'),
    #     nn.ReLU(),
    #     nn.Conv2d(12, 8, (3, 3), padding='same'),
    #     nn.ReLU(),
    #     nn.MaxPool2d((2, 2)),
    #     #
    #     # (N, 8, 16, 32)
    #     #
    #     nn.Conv2d(8, 16, (3, 3), padding='same'),
    #     nn.ReLU(),
    #     nn.Conv2d(16, 16, (3, 3), padding='same'),
    #     nn.ReLU(),
    #     nn.MaxPool2d((2, 2)),
    #     # (N, 16, 8, 16)
    #     #
    #     #
    #     nn.Conv2d(16, 32, (3, 3), padding='same'),
    #     nn.ReLU(),
    #     nn.Conv2d(32, 32, (3, 3), padding='same'),
    #     nn.ReLU(),
    #     nn.MaxPool2d((2, 2)),
    #     # (N, 32, 4, 8)
    #     #
    #     nn.Flatten(),
    #     #
    #     # (N, 32*32)
    #     #
    #     #
    #     nn.Linear(in_features=32 * 32, out_features=len(CLASSES))
    #     #
    # )
    return network


def print_output(data):
    # Calculate probabilities from logits
    data = nn.functional.softmax(data, dim=1)
    # Calculate maxima and labels
    max_scores, pred_labels = torch.max(data, dim=1)

    # Print output line by line
    for (label, score, probs) in zip(pred_labels, max_scores, data):
        probs = f"[ {', '.join([f'{prob:.2f}' for prob in probs])} ]"
        print(f"{probs} ... {label} ({CLASSES[label]}) [{score * 100:.2f}%]")


if __name__ == "__main__":
    print('CUDA available:', torch.cuda.is_available())
    train()
