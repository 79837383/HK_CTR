# -*- coding: utf-8 -*
import paddle.v2 as paddle
from paddle.v2 import layer
from paddle.v2 import data_type as dtype
from utils import logger, ModelType


class CTRmodel(object):
    '''
    A CTR model which implements wide && deep learning model.
    '''

    def __init__(self,
                 dnn_layer_dims,
                 dnn_input_dim,
                 lr_input_dim,
                 model_type=ModelType.create_classification(),
                 is_infer=False):
        '''
        @dnn_layer_dims: list of integer
            dims of each layer in dnn
        @dnn_input_dim: int
            size of dnn's input layer
        @lr_input_dim: int
            size of lr's input layer
        @is_infer: bool
            whether to build a infer model
        '''
        self.dnn_layer_dims = dnn_layer_dims #[128, 64, 32, 1]
        self.dnn_input_dim = dnn_input_dim #61
        self.lr_input_dim = lr_input_dim #10040001
        self.model_type = model_type #classification=0  regression=1
        self.is_infer = is_infer #true false

        self._declare_input_layers()

        self.dnn = self._build_dnn_submodel_(self.dnn_layer_dims)
        self.lr = self._build_lr_submodel_()

        # model's prediction
        # TODO(superjom) rename it to prediction
        if self.model_type.is_classification():
            self.model = self._build_classification_model(self.dnn, self.lr)
        if self.model_type.is_regression():
            self.model = self._build_regression_model(self.dnn, self.lr)

    def _declare_input_layers(self):
        self.dnn_merged_input = layer.data(
            name='dnn_input',
            #type    InputType(dim=61, seq_type=SequenceType.NO_SEQUENCE, type=DataType.SparseNonValue)
            # sparse_binary_vector 稀疏的01向量，即大部分值为0，但有值的地方必须为1
            type=paddle.data_type.sparse_binary_vector(self.dnn_input_dim)) #稀疏二进制向量  #dnn_input_dim #61

        self.lr_merged_input = layer.data(
            name='lr_input',
            #type    InputType(dim=10040001, seq_type=SequenceType.NO_SEQUENCE, type=DataType.SparseValue)
            type=paddle.data_type.sparse_float_vector(self.lr_input_dim)) #稀疏浮点向量

        if not self.is_infer:
            self.click = paddle.layer.data(
                name='click', type=dtype.dense_vector(1)) #dense_vector  稠密浮点向量

    def _build_dnn_submodel_(self, dnn_layer_dims):
        '''
        build DNN submodel.  # dnn_layer_dims = [128, 64, 32, 1]
        '''
        #dnn_merged_input是一个数据层，这个全连接层是128维的
        dnn_embedding = layer.fc(input=self.dnn_merged_input,
                                 size=dnn_layer_dims[0])
        _input_layer = dnn_embedding
        for i, dim in enumerate(dnn_layer_dims[1:]):  #64, 32, 1
            fc = layer.fc(input=_input_layer,
                          size=dim,
                          act=paddle.activation.Relu(),#ReLU activation
                          name='dnn-fc-%d' % i)
            _input_layer = fc
        return _input_layer

    def _build_lr_submodel_(self):
        '''
        config LR submodel
        '''
        fc = layer.fc(input=self.lr_merged_input,
                      size=1,
                      act=paddle.activation.Relu())
        return fc

    def _build_classification_model(self, dnn, lr):
        merge_layer = layer.concat(input=[dnn, lr])
        self.output = layer.fc(
            input=merge_layer,
            size=1,
            # use sigmoid function to approximate ctr rate, a float value between 0 and 1.
            act=paddle.activation.Sigmoid())

        if not self.is_infer: #如果不是预测就添加上label
            self.train_cost = paddle.layer.multi_binary_label_cross_entropy_cost( #交叉熵
                input=self.output, label=self.click)
        return self.output

    def _build_regression_model(self, dnn, lr):
        merge_layer = layer.concat(input=[dnn, lr])
        self.output = layer.fc(input=merge_layer,
                               size=1,
                               act=paddle.activation.Sigmoid())
        if not self.is_infer:
            self.train_cost = paddle.layer.square_error_cost( #求方差
                input=self.output, label=self.click)
        return self.output
