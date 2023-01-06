import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel
from matplotlib import pyplot as plt


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs = 50
    steps = 0
    running_loss = 0
    print_every = 50
    train_losses = []
    for e in range(epochs):
        model.train()
        for images, labels in train_set:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_losses.append(loss.item())

            if steps % print_every == 0:
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. ")
                running_loss = 0
            steps += 1
    
    # plot training curve
    plt.plot(train_losses, label='Training loss')
    plt.legend(frameon=False)
    plt.show()
    
    torch.save(model.state_dict(), 's1_development_environment/exercise_files/final_exercise/checkpoint.pth')
    print('Model saved to: s1_development_environment/exercise_files/final_exercise/checkpoint.pth')



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    _, test_set = mnist()

    with torch.no_grad():
        model.eval()
        accuracy = 0
        for images, labels in test_set:
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        accuracy /= len(test_set)
    print(f"Accuracy: {accuracy.item() * 100}%")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

  