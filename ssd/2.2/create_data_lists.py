from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(train_path='../../dataset/voc2012/train.txt',
                      valid_path='../../dataset/voc2012/valid.txt',
                      output_folder='./')
