from DataOps import DataManager
from multiprocessing import Process
import keyboard

def main():
    manager = DataManager("EuroSAT")
    manager.show_image()
    
    # train_results, test_results = manager.train(epochs=10, batch_size=100, lr = 0.01, scheduler=False)
    # manager.plot_training_results(train_results, test_results)
    manager.show_test_example()

if __name__ == "__main__":
    main()