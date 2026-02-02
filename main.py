import os
from DataOps import DataManager

def main():
    manager = DataManager("EuroSAT")
    manager.train(csv_name="test", lr=0.001)
    DataManager.plot_training_results(csv_name="02-02 16-08")
    return 0
    
if __name__ == "__main__":
    main()