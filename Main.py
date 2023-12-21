from MIST_Data import MISTDataClass



def main():
    mist = MISTDataClass()

    mist.prepare_data(True, True)

    #mist.show()
    mist.info()

    mist.training_model()
    mist.show_test_head_results()
    mist.test_model()


if __name__ == '__main__':
    main()
