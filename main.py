import os
from DataOps import DataManager


def main():
    manager = DataManager("EuroSAT")
    train_results, test_results = manager.train()
    manager.plot_training_results(train_results, test_results)
    return 0
    


if __name__ == "__main__":
    main()