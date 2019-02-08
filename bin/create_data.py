# create filtered and test data
# filtered by definition, see http://fhtagn.net/prog/2016/08/19/kaggle-uns.html
from utils.data.data import DataManager


def main():
    dm = DataManager()
    dm.create_cleaned_train_data()
    dm.create_test_data()


if __name__ == '__main__':
    main()
