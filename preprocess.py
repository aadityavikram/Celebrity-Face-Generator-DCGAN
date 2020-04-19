import time
import shutil
import zipfile
from PIL import Image
from os import path, makedirs, listdir, remove


def extract(source="data/img_align_celeba.zip"):
    if not path.exists(source):
        print('Dataset zip not found or already extracted')
    else:
        print('Dataset zip found. Extracting....')
        zip_file = source
        zip_ref = zipfile.ZipFile(zip_file, 'r')
        start = time.time()
        zip_ref.extractall(path="data")
        zip_ref.close()
        remove(zip_file)
        print('Extracted | Time elapsed --> {} seconds\n'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start))))


def preprocess(source="data/img_align_celeba/", dest="data/celeb/celeb", size=64):
    if not path.exists(dest):
        makedirs(dest)
    start = time.time()
    for i, file in enumerate(listdir(source)):
        img = Image.open(path.join(source, file))
        img = img.resize((size, size))
        img.save(path.join(dest, file))
        if i % 10000 == 0 and i is not 0:
            print('Preprocessed {} images'.format(i))
    print('Preprocessing done | Time elapsed --> {} seconds'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start))))


def split(source='data/celeb/celeb'):
    print('Splitting into train and test folders....')
    images_path = source
    count = len([files for files in listdir(images_path)])

    print('Total number of files --> {}'.format(count))

    train_images_path = path.join(source, 'train/data')
    test_images_path = path.join(source, 'test/data')

    test_count = count // 10
    train_count = count - test_count

    if not path.exists(train_images_path):
        makedirs(train_images_path)
    if not path.exists(test_images_path):
        makedirs(test_images_path)

    start = time.time()
    for i, file in enumerate(listdir(images_path)):
        if file == 'train' or file == 'test':
            continue
        src_file = path.join(images_path, file)
        if i < train_count:
            dst_file = path.join(train_images_path, file)
        else:
            dst_file = path.join(test_images_path, file)
        shutil.copyfile(src_file, dst_file)
        remove(src_file)
        if i % 10000 == 0 and i is not 0:
            print('Split {} images'.format(i))
    print('Spilt Done | Time elapsed --> {} seconds\n'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start))))


def balance(source='data/celeb/celeb', batch_size=32):
    train_dir = path.join(source, 'train/data')
    test_dir = path.join(source, 'test/data')
    len_train = len(list(listdir(train_dir)))
    len_test = len(list(listdir(test_dir)))
    len_train_new = (len_train // batch_size) * batch_size
    len_test_new = (len_test // batch_size) * batch_size

    print('Balancing dataset.... (N(images) % batch_size == 0)')
    start = time.time()
    print('N(train) = {} | N(test) = {} before balancing'.format(len_train, len_test))
    for i, file in enumerate(listdir(train_dir)):
        if i >= len_train_new:
            remove(path.join(train_dir, file))
    for i, file in enumerate(listdir(test_dir)):
        if i >= len_test_new:
            remove(path.join(test_dir, file))
    print('N(train) = {} | N(test) = {} after balancing'.format(len_train_new, len_test_new))
    print('N_Removed(train) = {} | N_Removed(test) = {}'.format(len_train - len_train_new, len_test - len_test_new))
    print('Balanced dataset | Time elapsed --> {} seconds'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start))))


def zip_dir(source='data/celeb', name='data/celeb'):
    print('Zipping the dataset....')
    start = time.time()
    shutil.make_archive(name, 'zip', source)
    print('Zipped | Time elapsed --> {} seconds'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start))))


def main():
    start = time.time()
    if not path.exists('data/img_align_celeba/'):
        extract()
    else:
        print('Dataset already extracted. Preprocessing, Splitting and Zipping the dataset....')
    preprocess(size=64)
    split()
    # balance()
    # zip_dir()
    print('All done | Time elapsed --> {} seconds'.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start))))


if __name__ == '__main__':
    main()
