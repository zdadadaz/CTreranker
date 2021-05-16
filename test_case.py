from test import datagen

def main():
    a = datagen.TestStringMethods()
    # a.test_train_dataloader()
    # a.test_test_dataloader()
    a.test_train_trec_dataloader()

if __name__ == '__main__':
    main()