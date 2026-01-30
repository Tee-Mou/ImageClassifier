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
from MNISTModel import MNISTModel
from EuroSATModel import EuroSATModel
from torch import nn
from tqdm import tqdm

class DataManager:

    model_list = {
        "MNIST": {
            "model": MNISTModel,
            "dataset": datasets.MNIST,
            "bonus_args": ["train"]
        }, 
        "EuroSAT": {
            "model": EuroSATModel, 
            "dataset": datasets.EuroSAT,
            "bonus_args": []
        }
    }

    def __init__(self, model : str | None = None, batch_size: int = 32, path: str | None = None) -> None:
        self.batch_size = batch_size
        self.load_model(model, path, True)
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)
        self.select_criterion(nn.BCEWithLogitsLoss)

    def load_model(self, model, name: str | None = None, load_data = False):
        self.model = self.model_list[model]["model"]()
        if name:
            self.load_model_state(name)

        dataset_args = {
            "root":"data",
            "download":True,
            "transform":ToTensor(),
        }
        if "train" in self.model_list[model]["bonus_args"]:
            dataset_args["train"] = True
            self.train_data = self.model_list[model]["dataset"](**dataset_args)
            dataset_args["train"] = False
            self.test_data = self.model_list[model]["dataset"](**dataset_args)
            self.labels = self.test_data.classes
        else:
            data = self.model_list[model]["dataset"](**dataset_args)
            self.train_data, self.test_data = random_split(data, [0.8, 0.2])
            self.labels = data.classes
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size)

    def save_model_state(self, name):
        path = "./model/" + self.model.__name__ + "/" + name + ".pth"
        torch.save(self.model.state_dict(), path)

    def load_model_state(self, name):
        path = "./model/" + self.model.__name__ + "/" + name + ".pth"
        self.model.load_state_dict(torch.load(path, weights_only=True))
    
    def show_image(self) -> None:
        datapoint = next(iter(self.test_loader))
        img = TF.to_pil_image(datapoint[0][0])
        plt.imshow(img)
        plt.title(f"Example Image of {datapoint[1][0].item()} in the Dataset")
        plt.show()
    
    def select_criterion(self, criterion):
        self.criterion = criterion()

    def train(self, epochs = 10, lr = 0.01, batch_size = 32, scheduler = True):
        scheduler_factor = [1, 0.1]
        best_test_loss = np.inf
        train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size)
        optimiser = optim.SGD(self.model.parameters(), lr = lr)
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
                outputs = self.model(images)
                train_loss = self.criterion(outputs, one_hot_targets.float())
                batch_loss = train_loss.item()
                pbar.desc = f"    Processing Training Batch {batch_id} | Batch Loss = {batch_loss}"

                optimiser.zero_grad()
                train_loss.backward()
                optimiser.step()

                batch_number = batch_id + epoch * len(train_loader)
                train_results.append((batch_number, batch_loss))       

            test_loss, test_accuracy = self.test()
            test_results.append(((epoch + 1) * len(train_loader), test_loss, test_accuracy))
            scheduler.step(test_accuracy)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                self.save_model_state("best")
            self.save_model_state("latest")

        return (train_results, test_results)

    def test(self, batch_size = 32):
        self.model.eval()
        test_loader = DataLoader(dataset=self.test_data, batch_size=batch_size)
        total_tests = len(self.test_data)
        test_accuracy = 0
        test_loss = 0
        for batch_id, (images, targets) in (
                enumerate(pbar := tqdm(test_loader, leave=False))
        ):
            with torch.no_grad():
                one_hot_targets = torch.nn.functional.one_hot(targets, 10)
                outputs = self.model(images)
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
    
    def show_test_example(self, count = 1):
        self.model.eval()
        with torch.no_grad():
            for _ in range(count):
                random_index = np.random.randint(0, self.batch_size)
                data = next(iter(self.test_loader)) 
                inp, target = (data[0][random_index], self.labels[data[1][random_index]])
                img = TF.to_pil_image(inp)
                inp = inp.unsqueeze(0)
                output = self.model(inp)
                prediction = self.labels[output.argmax()]
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





            
