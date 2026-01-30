import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, utils
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF 
import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import MNISTModel
from torch import nn
from tqdm import tqdm



class DataManager:

    def load_mnist(self, root: str = "data") -> None:
        self.train_data = datasets.MNIST(
            root="data",
            download=True,
            train=True,
            transform=ToTensor(),
        )
        self.test_data = datasets.MNIST(
            root="data",
            download=True,
            train=False,
            transform=ToTensor(),
        )
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)
        self.select_criterion(nn.BCEWithLogitsLoss)
    
    def show_image(self) -> None:
        plt.imshow(self.test_data.data[0], cmap="gray")
        plt.title(f"Example Image of {self.test_data.targets[0].item()} in the MNIST Dataset")
        plt.show()

    def fetch_example(self) -> tuple[torch.FloatTensor, int]:
        img, label = self.test_data[0]
        return (img, label)
    
    def select_criterion(self, criterion):
        self.criterion = criterion()

    def train(self, model: MNISTModel, epochs = 10, lr = 0.01, batch_size = 32, scheduler = True):
        scheduler_factor = [1, 0.1]
        best_test_loss = np.inf
        train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size)
        optimiser = optim.SGD(model.parameters(), lr = lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimiser,patience=1,mode="max",threshold=0.001, factor=scheduler_factor[scheduler])

        train_results = []
        test_results = []

        for epoch in (pbar_epoch := tqdm(range(epochs))):
            try:
                previous_accuracy = test_results[-1][2]
                previous_epoch = test_results[-1][0]
            except IndexError:
                previous_epoch = "N/A"
                previous_accuracy = 0
            pbar_epoch.set_description(
                f"Processing epoch {epoch + 1} | Epoch {previous_epoch} Accuracy: {previous_accuracy} | lr: {scheduler.get_last_lr()}"
            )
            for batch_id, (images, targets) in (
                enumerate(pbar := tqdm(train_loader, leave=False))
            ):
                one_hot_targets = torch.nn.functional.one_hot(targets, 10)
                outputs = model(images)
                train_loss = self.criterion(outputs, one_hot_targets.float())
                batch_loss = train_loss.item()
                pbar.desc = f"    Processing Training Batch {batch_id} | Batch Loss = {batch_loss}"

                optimiser.zero_grad()
                train_loss.backward()
                optimiser.step()

                batch_number = batch_id + epoch * len(train_loader)
                train_results.append((batch_number, batch_loss))       

            test_loss, test_accuracy = self.test(model=model)
            test_results.append(((epoch + 1) * len(train_loader), test_loss, test_accuracy))
            scheduler.step(test_accuracy)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), "./model/best.pth")
            torch.save(model.state_dict(), "./model/latest.pth")

        return (train_results, test_results)

    def test(self, model: MNISTModel, batch_size = 32):
        model.eval()
        test_loader = DataLoader(dataset=self.test_data, batch_size=batch_size)
        total_tests = len(self.test_data)
        test_accuracy = 0
        test_loss = 0
        for batch_id, (images, targets) in (
                enumerate(pbar := tqdm(test_loader, leave=False))
        ):
            with torch.no_grad():
                one_hot_targets = torch.nn.functional.one_hot(targets, 10)
                outputs = model(images)
                predictions = outputs.argmax(1)
                batch_loss = self.criterion(outputs, one_hot_targets.float()).item()
                test_loss += batch_loss
                for i in range(targets.size(0)):
                    if targets[i] == predictions[i]:
                        test_accuracy += 1
            pbar.desc = f"Processing Test Batch {batch_id} | Batch Loss = {batch_loss}"
        test_accuracy /= total_tests
        test_loss /= total_tests
        return test_loss, test_accuracy
    
    def show_test_examples(self, model: MNISTModel, count = 1):
        model.eval()
        with torch.no_grad():
            for _ in range(count):
                random_index = np.random.randint(0, len(self.test_data))
                inp, target = self.test_data[random_index]
                img = TF.to_pil_image(inp)
                inp = inp.unsqueeze(0)
                output = model(inp)
                prediction = output.argmax()
                plt.imshow(img)
                plt.title(f"Predicted: {prediction} | Actual {target}")
                plt.show()
            
    @staticmethod
    def plot_training_results(train_results, test_results):
        
        fig, ax1 = plt.subplots()
        x_train, y_train = zip(*train_results)
        x_test, y_test, acc_test = zip(*test_results)
        ax1.set_xlabel("Batch Number")
        ax1.set_ylabel("Training Loss", color="red")
        ax1.plot(x_train, y_train, color="red")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Test Loss", color="blue")
        ax2.plot(x_test, y_test, color="blue")

        plt.title("Training and Test Loss for MNIST Model Training")
        fig.tight_layout()
        plt.show()

        plt.ylabel("Test Accuracy")
        plt.xlabel("Batch Number")
        plt.plot(x_test, acc_test, color="orange")
        plt.show()





            
