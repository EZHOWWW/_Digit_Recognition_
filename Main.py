from MIST_Data import MISTDataClass
from LoadMyData import get_my_data


def main():
    mist = MISTDataClass(get_my_data('My_data'))

    mist.prepare_data()

    mist.show()
    mist.info()

    mist.training_model()
    mist.test_model()
    mist.show_test_head_results()


if __name__ == '__main__':
    main()
