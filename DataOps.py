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
import pandas

from Evaluator import Evaluator as e

class DataManager:

    model_list = {
        "MNIST": {
            "model": MNISTModel,
            "dataset": datasets.MNIST,
            "bonus_args": ["train"],
            "eval": {"Best Only": e.best_only}
        }, 
        "EuroSAT": {
            "model": EuroSATModel, 
            "dataset": datasets.EuroSAT,
            "bonus_args": [],
            "eval": {"Best Two": e.best_two, "Best Only": e.best_only}
        }
    }

    def __init__(self, model : str, batch_size: int = 32, name: str | None = None) -> None:
        self.batch_size = batch_size
        self.load_model(model, name, True)
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)
        self.select_criterion(nn.CrossEntropyLoss)
        self.select_optimiser(optim.AdamW)

    def load_model(self, model: str, name: str | None = None, load_data = False):
        self.name = model
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
        path = "./model/" + self.name + "/" + name + ".pth"
        torch.save(self.model.state_dict(), path)

    def load_model_state(self, name):
        path = "./model/" + self.name + "/" + name + ".pth"
        self.model.load_state_dict(torch.load(path, weights_only=True))
    
    def show_image(self) -> None:
        datapoint = next(iter(self.test_loader))
        img = TF.to_pil_image(datapoint[0][0])
        plt.imshow(img)
        plt.title(f"Example Image of {self.labels[datapoint[1][0].item()]} in the Dataset")
        plt.show()
    
    def select_optimiser(self, optimiser):
        self.optimiser = optimiser

    def select_criterion(self, criterion):
        self.criterion = criterion()

    def train(self, csv_name, epochs = 10, lr = 0.01, scheduler = True):
        best_test_loss = np.inf
        train_loader = DataLoader(dataset=self.train_data, batch_size=self.batch_size)
        optimiser = self.optimiser(self.model.parameters(), lr = lr, weight_decay=0.24)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimiser,mode="min",threshold=0.001, factor=0.01)
        
        csv_path = "./results/{}".format(csv_name)
        model_results: pandas.DataFrame = pandas.DataFrame(columns=["Training Loss", "Test Loss"] + list(self.model_list[self.name]["eval"]))
        model_results.index.name="Batch No."

        current_batch = 0

        for epoch in (pbar_epoch := tqdm(range(epochs))):
            try:
                previous_epoch = epoch
                previous_loss = model_results.iloc[current_batch]["Test Loss"]
            except KeyError, IndexError:
                previous_epoch = "N/A"
                previous_loss = 0
            pbar_epoch.set_description(
                f"Processing epoch {epoch + 1} | Epoch {previous_epoch} Loss: {previous_loss} | lr: {scheduler.get_last_lr()}"
            )
            for batch_id, (images, targets) in (
                enumerate(pbar := tqdm(train_loader, leave=False))
            ):
                current_batch += 1
                one_hot_targets = torch.nn.functional.one_hot(targets, 10)
                outputs = self.model(images)
                train_loss = self.criterion(outputs, one_hot_targets.float())
                batch_loss = train_loss.item()
                pbar.desc = f"    Processing Training Batch {batch_id} | Batch Loss = {batch_loss}"

                optimiser.zero_grad()
                train_loss.backward()
                optimiser.step()

                model_results.loc[current_batch] = [batch_loss, None, None, None]

            test_loss, test_accuracy = self.test()
            model_results.at[current_batch, "Test Loss"] = test_loss
            for k, v in test_accuracy.items():
                model_results.at[current_batch, k] = v

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                self.save_model_state("best")
            if scheduler:
                scheduler.step(test_loss)
            model_results.to_csv(csv_path)
            self.save_model_state("latest")

    def test(self):
        self.model.eval()
        test_loader = DataLoader(dataset=self.test_data, batch_size=self.batch_size)
        total_tests = len(self.test_data)
        test_accuracy = dict((k, 0) for k in self.model_list[self.name]["eval"])
        test_loss = 0
        for batch_id, (images, targets) in (
                enumerate(pbar := tqdm(test_loader, leave=False))
        ):
            with torch.no_grad():
                one_hot_targets = torch.nn.functional.one_hot(targets, 10)
                outputs = self.model(images)
                batch_loss = self.criterion(outputs, one_hot_targets.float()).item()
                test_loss += batch_loss
                for eval_method, f in self.model_list[self.name]["eval"].items():
                    test_accuracy[eval_method] += f(outputs, targets) / total_tests
            pbar.desc = f"Processing Test Batch {batch_id} | Batch Loss = {batch_loss}"
        test_loss /= len(test_loader)
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
                prediction_order = output.argsort(dim=1, descending=True)[0]
                prediction = self.labels[prediction_order[0]]
                prediction_backup = self.labels[prediction_order[1]]
                plt.imshow(img)
                plt.title(f"Predicted: {prediction} (Second: {prediction_backup}) | Actual: {target}")
                plt.show()
            
    @staticmethod
    def plot_training_results(csv_path = None, csv_name = None):
        if csv_path == None:
            csv_path = "./results/{}".format(csv_name)
        data = pandas.read_csv(csv_path)
        for col in data.columns:
            if col == "Batch No.":
                continue
            col_data = data[col].dropna()
            col_data.plot()
        plt.legend()
        plt.show()




            
