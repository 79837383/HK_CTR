# -*- coding: utf-8 -*
import sys
import csv
import cPickle
import argparse
import os
import numpy as np

from utils import logger, TaskMode

parser = argparse.ArgumentParser(description="PaddlePaddle CTR example")

#训练数据集
# --data_path train.txt
parser.add_argument(
    '--data_path', type=str, required=True, help="path of the Avazu dataset")

#演示数据输出文件
# --output_dir output
parser.add_argument(
    '--output_dir', type=str, required=True, help="directory to output")

#预先扫描数据生成ID的个数，这里是扫描的文件行数
# --num_lines_to_detect 1000
parser.add_argument(
    '--num_lines_to_detect',
    type=int,
    default=500000,
    help="number of records to detect dataset's meta info")

#生成测试集的行数
# --test_set_size 100
parser.add_argument(
    '--test_set_size',
    type=int,
    default=10000,
    help="size of the validation dataset(default: 10000)")

#生成训练集的行数
# --batch_size 100
parser.add_argument(
    '--train_size',
    type=int,
    default=100000,
    help="size of the trainset (default: 100000)")

args = parser.parse_args()
'''
The fields of the dataset are:

    0. id: ad identifier
    1. click: 0/1 for non-click/click
    2. hour: format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC.
    3. C1 -- anonymized categorical variable
    4. banner_pos
    5. site_id
    6. site_domain
    7. site_category
    8. app_id
    9. app_domain
    10. app_category
    11. device_id
    12. device_ip
    13. device_model
    14. device_type
    15. device_conn_type
    16. C14-C21 -- anonymized categorical variables

We will treat the following fields as categorical features:

    - C1
    - banner_pos
    - site_category
    - app_category
    - device_type
    - device_conn_type

and some other features as id features:

    - id
    - site_id
    - app_id
    - device_id

The `hour` field will be treated as a continuous feature and will be transformed
to one-hot representation which has 24 bits.

This script will output 3 files:

1. train.txt
2. test.txt
3. infer.txt

all the files are for demo.
'''

feature_dims = {}

categorial_features = (
    'C1 banner_pos site_category app_category ' + 'device_type device_conn_type'
).split()
#[5, 3, 11, 8, 5, 4, 24]


id_features = 'id site_id app_id device_id _device_id_cross_site_id'.split()


def get_all_field_names(mode=0):
    '''
    @mode: int
        0 for train, 1 for test
    @return: list of str
    '''
    return categorial_features + ['hour'] + id_features + ['click'] \
        if mode == 0 else []


class CategoryFeatureGenerator(object):
    '''
    Generator category features.

    Register all records by calling `register` first, then call `gen` to generate
    one-hot representation for a record.
    '''

    def __init__(self):
        self.dic = {'unk': 0}
        self.counter = 1

    #获取数据集中同一个属性有多少种不同的值，并且将原来的取值替换为我们自己规定的取值例如：
    #('C1', {'1005': 1, 'unk': 0, '1010': 3, '1002': 2, '1001': 4})
    def register(self, key):
        '''
        Register record.
        '''
        if key not in self.dic:
            self.dic[key] = self.counter
            self.counter += 1

    def size(self):
        return len(self.dic)

    def gen(self, key):
        '''
        Generate one-hot representation for a record.
        '''
        if key not in self.dic:
            res = self.dic['unk']
        else:
            res = self.dic[key]
        return [res]

    def __repr__(self):
        return '<CategoryFeatureGenerator %d>' % len(self.dic)


class IDfeatureGenerator(object):
    def __init__(self, max_dim, cross_fea0=None, cross_fea1=None):
        '''
        @max_dim: int
            Size of the id elements' space
        '''
        self.max_dim = max_dim
        self.cross_fea0 = cross_fea0
        self.cross_fea1 = cross_fea1

    def gen(self, key):
        '''
        Generate one-hot representation for records
        '''
        return [hash(key) % self.max_dim]

    def gen_cross_fea(self, fea1, fea2):
        key = str(fea1) + str(fea2)
        return self.gen(key)

    def size(self):
        return self.max_dim


class ContinuousFeatureGenerator(object):
    def __init__(self, n_intervals):
        self.min = sys.maxint
        self.max = sys.minint
        self.n_intervals = n_intervals

    def register(self, val):
        self.min = min(self.minint, val)
        self.max = max(self.maxint, val)

    def gen(self, val):
        self.len_part = (self.max - self.min) / self.n_intervals
        return (val - self.min) / self.len_part


# init all feature generators
fields = {}
#categorial_features = ['C1', 'banner_pos', 'site_category', 'app_category', 'device_type', 'device_conn_type']
for key in categorial_features:
    fields[key] = CategoryFeatureGenerator()
'''fields:,
{'app_category': <CategoryFeatureGenerator 1>, 
'site_category': <CategoryFeatureGenerator 1>, 
'device_type': <CategoryFeatureGenerator 1>, 
'banner_pos': <CategoryFeatureGenerator 1>, 
'device_conn_type': <CategoryFeatureGenerator 1>, 
'C1': <CategoryFeatureGenerator 1>}'''

'''"id_features",['id', 'site_id', 'app_id', 'device_id', '_device_id_cross_site_id']'''
for key in id_features:
    # for cross features
    if 'cross' in key:
        feas = key[1:].split('_cross_')
        #"feas",['device_id', 'site_id']
        fields[key] = IDfeatureGenerator(10000000, *feas) #*feas = "site_id"
    # for normal ID features
    else:
        fields[key] = IDfeatureGenerator(10000)
'''fields,
{'app_category': <CategoryFeatureGenerator 8>, 
'site_category': <CategoryFeatureGenerator 11>, 
'device_type': <CategoryFeatureGenerator 5>, 
'banner_pos': <CategoryFeatureGenerator 3>, 
'site_id': <__main__.IDfeatureGenerator object at 0x10ed7e990>, 
'app_id': <__main__.IDfeatureGenerator object at 0x10ed7e9d0>, 
'_device_id_cross_site_id': <__main__.IDfeatureGenerator object at 0x10ed7ea50>, 
'device_conn_type': <CategoryFeatureGenerator 4>, 
'C1': <CategoryFeatureGenerator 5>, 
'id': <__main__.IDfeatureGenerator object at 0x10ed7e950>, 
'device_id': <__main__.IDfeatureGenerator object at 0x10ed7ea10>}

Feature	Dimention
id	10000
site_id	10000
app_id	10000
device_id	10000
device_id X site_id	1000000

'''

# used as feed_dict in PaddlePaddle
field_index = dict((key, id)
                   for id, key in enumerate(['dnn_input', 'lr_input', 'click']))
#"field_index",{'dnn_input': 0, 'click': 2, 'lr_input': 1}


#--path ./res/train.txt
#--topn 1000
def detect_dataset(path, topn, id_fea_space=10000):
    '''
    Parse the first `topn` records to collect meta information of this dataset.

    NOTE the records should be randomly shuffled first.
    '''
    # create categorical statis objects.
    logger.warning('detecting dataset')

    with open(path, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)

        for row_id, row in enumerate(reader):
            # if row_id < 10:
            #     print "____________"
            #     print row
            if row_id > topn:
                break

            for key in categorial_features:
                fields[key].register(row[key])

        # for key in categorial_features:
        #     #print (key,fields[key].counter)
        #     print (key,fields[key].dic)
        # exit(0)
        #
        # ('C1', 5)
        # ('banner_pos', 3)
        # ('site_category', 11)
        # ('app_category', 8)
        # ('device_type', 5)
        # ('device_conn_type', 4)

        # ('C1', {'1005': 1, 'unk': 0, '1010': 3, '1002': 2, '1001': 4})
        # ('banner_pos', {'1': 2, '0': 1, 'unk': 0})
        # ('site_category',
        #  {'28905ebd': 1, '75fa27f6': 10, '76b2941d': 6, '72722551': 9, 'f028772b': 3, '0569f928': 2, '3e814130': 5,
        #   '50e219e0': 4, '335d28a8': 8, 'f66779e6': 7, 'unk': 0})
        # ('app_category',
        #  {'4ce2e9fc': 7, '07d7df22': 1, 'cef3e649': 3, '75d80bbe': 6, '8ded1f7a': 4, '0f2161f8': 2, 'f95efa07': 5,
        #   'unk': 0})
        # ('device_type', {'1': 1, '0': 2, 'unk': 0, '5': 4, '4': 3})
        # ('device_conn_type', {'2': 1, '3': 3, 'unk': 0, '0': 2})

    for key, item in fields.items():
        feature_dims[key] = item.size()
        #print (key,item.size())

    feature_dims['hour'] = 24
    feature_dims['click'] = 1

    feature_dims['dnn_input'] = np.sum(
        feature_dims[key] for key in categorial_features + ['hour']) + 1
    feature_dims['lr_input'] = np.sum(feature_dims[key]
                                      for key in id_features) + 1
    #print(feature_dims['dnn_input'],feature_dims['lr_input']) #(61, 10040001)
    return feature_dims


def load_data_meta(meta_path):
    '''
    Load dataset's meta infomation.
    '''
    feature_dims, fields = cPickle.load(open(meta_path, 'rb'))
    return feature_dims, fields


def concat_sparse_vectors(inputs, dims):
    '''
    Concaterate more than one sparse vectors into one.

    @inputs: list
        list of sparse vector
    @dims: list of int
        dimention of each sparse vector
    '''
    res = []
    assert len(inputs) == len(dims)
    start = 0
    for no, vec in enumerate(inputs):
        for v in vec:
            res.append(v + start)
            # print v,start
        start += dims[no]

        #print "dims[no]",dims[no]
    # print dims
    # print inputs
    # print res
    # [5, 3, 11, 8, 5, 4, 24]
    # [[1], [2], [3], [1], [1], [2], [0]]
    # [1, 7, 11, 20, 28, 34, 36]
    # exit(0)
    return res

#--data_path train.txt --test_set_size 100
class AvazuDataset(object):
    '''
    Load AVAZU dataset as train set.
    '''

    def __init__(self,
                 train_path,
                 n_records_as_test=-1,
                 fields=None,
                 feature_dims=None):
        self.train_path = train_path
        self.n_records_as_test = n_records_as_test
        self.fields = fields
        # default is train mode. 静态方法
        self.mode = TaskMode.create_train()

        self.categorial_dims = [
            feature_dims[key] for key in categorial_features + ['hour']
        ]
        self.id_dims = [feature_dims[key] for key in id_features]

        #print (self.categorial_dims,self.id_dims) #([5, 3, 11, 8, 5, 4, 24], [10000, 10000, 10000, 10000, 10000000])

    def train(self):
        '''
        Load trainset.
        '''
        logger.info("load trainset from %s" % self.train_path)
        self.mode = TaskMode.create_train()
        with open(self.train_path) as f:
            reader = csv.DictReader(f)

            for row_id, row in enumerate(reader):
                # skip top n lines
                if self.n_records_as_test > 0 and row_id < self.n_records_as_test:
                    continue

                rcd = self._parse_record(row)
                if rcd:
                    yield rcd

    def test(self):
        '''
        Load testset.
        '''
        logger.info("load testset from %s" % self.train_path)
        self.mode = TaskMode.create_test()
        with open(self.train_path) as f:
            reader = csv.DictReader(f)

            for row_id, row in enumerate(reader):
                # skip top n lines
                if self.n_records_as_test > 0 and row_id > self.n_records_as_test:
                    break

                rcd = self._parse_record(row)
                if rcd:
                    yield rcd

    def infer(self):
        '''
        Load inferset.
        '''
        logger.info("load inferset from %s" % self.train_path)
        self.mode = TaskMode.create_infer()
        with open(self.train_path) as f:
            reader = csv.DictReader(f)

            for row_id, row in enumerate(reader):
                rcd = self._parse_record(row)
                if rcd:
                    yield rcd

    def _parse_record(self, row):
        '''
        Parse a CSV row and get a record.
        '''
        record = []
        # nInd = 0
        for key in categorial_features:
            record.append(self.fields[key].gen(row[key]))
            # if nInd < 10:
            #     nInd += 1
            #     print ("row[key]:",row[key])
            #     print (key,self.fields[key].gen(row[key]))
            #print (key,self.fields[key].gen(row[key]))

        record.append([int(row['hour'][-2:])])
        #print row['hour']  #14102100
        # print ([int(row['hour'][-2:])]) #???
        #print record  # [[1], [2], [3], [1], [1], [2], [0]]
        dense_input = concat_sparse_vectors(record, self.categorial_dims)
        #把源数据替换成 我们自己的类别属性对应的值，例如 "C1"之前和现在对应的值是(1001--4,1002--2,1005--1,1010--3)

        record = []
        for key in id_features:
            if 'cross' not in key:
                record.append(self.fields[key].gen(row[key]))
            else:
                fea0 = self.fields[key].cross_fea0
                fea1 = self.fields[key].cross_fea1
                record.append(self.fields[key].gen_cross_fea(row[fea0], row[fea1]))
        #print ("record",record)
        sparse_input = concat_sparse_vectors(record, self.id_dims)
        #print sparse_input
        record = [dense_input, sparse_input]

        if not self.mode.is_infer():
            record.append(list((int(row['click']), )))
        #print record #[[1, 7, 11, 20, 28, 34, 36], [5804, 14607, 21244, 34453, 122530], [0]]

        return record


def ids2dense(vec, dim):
    return vec


def ids2sparse(vec):
    return ["%d:1" % x for x in vec]

#--data_path ./res/train.csv
#--num_lines_to_detect 1000
detect_dataset(args.data_path, args.num_lines_to_detect)
#after detect_dataset(..,..),feature_dims存储每个属性的维度

# print CategoryFeatureGenerator()
# print IDfeatureGenerator(10000000, *feas)
#
# exit(0)


#Avazu广告公司
#--data_path train.txt --test_set_size 100
dataset = AvazuDataset(
    args.data_path,
    args.test_set_size,
    fields=fields,
    feature_dims=feature_dims)

#args.output_dir，arg2组合路径
output_trainset_path = os.path.join(args.output_dir, 'train.txt')
output_testset_path = os.path.join(args.output_dir, 'test.txt')
output_infer_path = os.path.join(args.output_dir, 'infer.txt')
output_meta_path = os.path.join(args.output_dir, 'data.meta.txt')

#在此案例中test.txt和infer.txt都是取的同一块的数据，只是infer.txt不带click标记,都是--test_set_size 100条

#('output/train.txt', 'output/test.txt', 'output/infer.txt', 'output/data.meta.txt')
#print (output_trainset_path,output_testset_path,output_infer_path,output_meta_path)

with open(output_trainset_path, 'w') as f:
    for id, record in enumerate(dataset.train()):
        if id and id % 10000 == 0:
            logger.info("load %d records" % id)
        if id > args.train_size:
            break
        dnn_input, lr_input, click = record

        dnn_input = ids2dense(dnn_input, feature_dims['dnn_input']) #arg1,61
        lr_input = ids2sparse(lr_input)
        line = "%s\t%s\t%d\n" % (' '.join(map(str, dnn_input)),
                                 ' '.join(map(str, lr_input)), click[0])

        #print line #1 7 11 20 28 34 36	5804:1 14607:1 21244:1 34453:1 122530:1	0

        f.write(line)
    logger.info('write to %s' % output_trainset_path)

with open(output_testset_path, 'w') as f:
    for id, record in enumerate(dataset.test()):
        dnn_input, lr_input, click = record
        dnn_input = ids2dense(dnn_input, feature_dims['dnn_input'])
        lr_input = ids2sparse(lr_input)
        line = "%s\t%s\t%d\n" % (' '.join(map(str, dnn_input)),
                                 ' '.join(map(str, lr_input)), click[0])
        f.write(line)
    logger.info('write to %s' % output_testset_path)

with open(output_infer_path, 'w') as f:
    for id, record in enumerate(dataset.infer()):
        dnn_input, lr_input = record
        dnn_input = ids2dense(dnn_input, feature_dims['dnn_input'])
        lr_input = ids2sparse(lr_input)
        line = "%s\t%s\n" % (
            ' '.join(map(str, dnn_input)),
            ' '.join(map(str, lr_input)), )
        f.write(line)
        if id > args.test_set_size:
            break
    logger.info('write to %s' % output_infer_path)

with open(output_meta_path, 'w') as f:
    lines = [
        "dnn_input_dim: %d" % feature_dims['dnn_input'],
        "lr_input_dim: %d" % feature_dims['lr_input']
    ]
    f.write('\n'.join(lines))
    logger.info('write data meta into %s' % output_meta_path)
