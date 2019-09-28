import numpy as np
import csv
import os
import numpy as np

train_file = './train.csv'
val_file = './val.csv'
test_file = './test6.csv'



trainper=0.9
valper=0.1

def write_csv_file(dataset, path):
    try:
        with open(path, 'w') as csv_file:
            for i in range(len(dataset)):
                line = dataset[i]
                csv_file.write(line + '\n')
    except Exception as e:
        print("Write a CSV file to path : %s, Case: %s" % (path, e))


if __name__ == '__main__':
    trainset,testset,valset=[],[],[]

    count0, count1, count2 = 0, 0, 0
    data=[]
    for i in range(5166):
        data.append(str(i))
    #np.random.shuffle(data)
    data=np.array(data)

    #trainset = data[0:int(len(data) * trainper)]
    #valset=data[int(len(data) * trainper):]
    # valset = data[int(len(data) * trainper):]
    #
    #
    #write_csv_file(trainset, train_file)
    #write_csv_file(valset, val_file)


    testset = data[0:]


    write_csv_file(testset, test_file)
