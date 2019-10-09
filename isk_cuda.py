import numpy as np
import time
import random
import os

args_file = 'bin/args.txt'
model_file = 'bin/model.csv'
input_file = 'bin/input.csv'
output_file = 'bin/output.csv'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # choose the available GPU


def save_model(psi: int, num: int, grids: np.ndarray):
    assert grids.shape[0] == psi*num;
    try:
        with open(file=args_file, mode='w', encoding='utf-8') as f:
            f.write(str(psi) + '\t' + str(num) + '\n')
        print('Save args to ' + args_file)
        save_matrix(model_file, grids)
        print('Save model to ' + model_file)
    except IOError as e:
        print('IO Error in saving model')


def save_matrix(file_name: str, matrix: np.ndarray):
    assert len(matrix.shape) == 2
    height, width = matrix.shape
    with open(file=file_name, mode='w', encoding='utf-8') as f:
        f.write(str(height) + '\t' + str(width) + '\n')
        for row in matrix:
            for ele in row:
                f.write(str(ele) + '\t')
            f.write('\n')


def load_matrix(file_name: str) -> np.ndarray:
    with open(file_name, mode='r', encoding='utf-8') as f:
        lines = f.readlines()
        height, width = map(int, lines[0].split())
        ret = np.zeros(shape=(height, width))
        lines.pop(0)
        for i in range(height):
            ret[i] = list(map(float, lines[i].split()))
        assert ret.shape[0] == height
    return ret


def load_model() -> (int, int, np.ndarray):
    psi, num = 0, 0
    grids = []
    try:
        with open(file=args_file, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            psi, num = map(int, lines[0].split())
    except FileNotFoundError as e:
        print(args_file + ' not found')
        exit()
    try:
        grids = load_matrix(model_file)
        assert grids.shape[0] == psi*num
    except FileNotFoundError as e:
        print(model_file + 'not found')
        exit()
    return psi, num, grids


def train_voronoi(x: np.ndarray, psi: int, num: int):
    grids = np.zeros((0, x.shape[1]))
    for i in range(num):
        selected_idx = random.sample(population=range(x.shape[0]), k=psi)
        grids = np.concatenate((grids, x[selected_idx]), axis=0)
    save_model(psi, num, grids)


def cal_features(x: np.ndarray) -> np.ndarray:
    save_matrix(input_file, x)
    try:
        os.system('nvcc main.cu -Wno-deprecated-gpu-targets -O3 -o main')
        os.system('./main ' + args_file + ' ' + model_file + ' ' +
                  input_file + ' ' + output_file)
    except FileNotFoundError:
        print('File Missing in compiling and executing')
    return load_matrix(output_file)


def cal_features_numpy(x: np.ndarray) -> np.ndarray:
    psi, num, grid = load_model()
    assert grid.shape[0] == psi*num
    assert x.shape[1] == grid.shape[1]
    ret = np.zeros((x.shape[0], num))
    for i in range(x.shape[0]):
        distance = np.sum(a=((grid - x[i])**2), axis=1).reshape((num, psi))
        ret[i] = np.argmin(a=distance, axis=1)
    return ret


if __name__ == '__main__':
    random_array = np.random.random(size=(1000000, 100))
    psi = 128
    num = 100
    train_voronoi(random_array, psi, num)
    x_idx = random.sample(population=range(random_array.shape[0]), k=100000)
    x = random_array[x_idx]
    print('Start to calculate using cuda')
    s1 = time.time()
    y = cal_features(x)
    e1 = time.time()
    print('cuda cost: ', e1 - s1)
    
    print('Start to calculate using numpy')
    s2 = time.time()
    y_numpy = cal_features_numpy(x)
    e2 = time.time()
    print('numpy cost: ', e2 - s2)
    if (y == y_numpy).all():
        print('Correct')
    else:
        print('Wrong: inconsistent in ', (y != y_numpy).nonzero()[0].shape[0], ' positions')


