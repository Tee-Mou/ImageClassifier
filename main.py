from DataOps import DataManager

def main():
    manager = DataManager(model="EuroSAT", name="EuroSAT97", batch_size=32)
    manager.show_test_example(5)
    return 0
    
if __name__ == "__main__":
    main()