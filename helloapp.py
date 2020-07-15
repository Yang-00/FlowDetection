#author:王旭，李姝洋，齐雨欣，段文思，李雨昕
#create: time: 2020-07-08
#update time: 2020-07-11

import zipfile
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import matplotlib.image as mping
from PIL import Image
import json
import numpy as np
import cv2
import sys
import time
# import scipy.io as io
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
from matplotlib import cm as CM
from paddle.utils.plot import Ploter

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear, Conv2DTranspose
from paddle.fluid.dygraph.base import to_variable
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time

from datetime import timedelta
import numpy as np
from PIL import Image
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import zipfile
import numpy as np
from PIL import Image
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import zipfile
# 定义卷积批归一化块
import cv2
import glob
import os
from datetime import datetime

from flask import Flask
from time import sleep
from gevent import monkey
import time

#段文思-part1-7.14-开始
from functools import wraps
from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_, or_
#段文思-part1-7.14-结束

monkey.patch_all()

# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp','mp4'])
ALLOWED_EXTENSIONS_VEDIO = set(['mp4'])

#深度神经网络
class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 param_attr=fluid.initializer.Xavier(uniform=False)):
        """
        name_scope, 模块的名字
        num_channels, 卷积层的输入通道数
        num_filters, 卷积层的输出通道数
        stride, 卷积层的步幅
        groups, 分组卷积的组数，默认groups=1不使用分组卷积
        act, 激活函数类型，默认act=None不使用激活函数
        """
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False,
            param_attr=param_attr)

        # 创建BatchNorm层
        self._batch_norm = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y
class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        """
        name_scope, 模块的名字
        num_channels, 卷积层的输入通道数
        num_filters, 卷积层的输出通道数
        stride, 卷积层的步幅
        groups, 分组卷积的组数，默认groups=1不使用分组卷积
        act, 激活函数类型，默认act=None不使用激活函数
        """
        super(ConvBNLayer, self).__init__()

        # 创建卷积层
        self._conv = fluid.dygraph.Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=None,
            bias_attr=False)

        # 创建BatchNorm层
        self._batch_norm = fluid.dygraph.BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y
class crowdnet(fluid.dygraph.Layer):
    '''
    网络
    crowdnet
    '''

    def __init__(self):
        super(crowdnet, self).__init__()
        # 定义deepnetwork
        self.conv01_1 = ConvBNLayer(num_channels=3, num_filters=64, filter_size=3, padding=1, act="relu")
        self.conv01_2 = ConvBNLayer(num_channels=64, num_filters=64, filter_size=3, padding=1, act="relu")

        self.pool01 = fluid.dygraph.Pool2D(pool_size=2, pool_type='max', pool_stride=2, pool_padding=0)

        self.conv02_1 = ConvBNLayer(num_channels=64, num_filters=128, filter_size=3, padding=1, act="relu")
        self.conv02_2 = ConvBNLayer(num_channels=128, num_filters=128, filter_size=3, padding=1, act="relu")
        self.pool02 = fluid.dygraph.Pool2D(pool_size=2, pool_type='max', pool_stride=2, pool_padding=0)

        self.conv03_1 = ConvBNLayer(num_channels=128, num_filters=256, filter_size=3, padding=1, act="relu")
        self.conv03_2 = ConvBNLayer(num_channels=256, num_filters=256, filter_size=3, padding=1, act="relu")
        self.conv03_3 = ConvBNLayer(num_channels=256, num_filters=256, filter_size=3, padding=1, act="relu")
        self.pool03 = fluid.dygraph.Pool2D(pool_size=2, pool_type='max', pool_stride=2, pool_padding=0)
        # (x-2/2+1)
        self.conv04_1 = ConvBNLayer(num_channels=256, num_filters=512, filter_size=3, padding=1, act="relu")
        self.conv04_2 = ConvBNLayer(num_channels=512, num_filters=512, filter_size=3, padding=1, act="relu")
        self.conv04_3 = ConvBNLayer(num_channels=512, num_filters=512, filter_size=3, padding=1, act="relu")
        self.pool04 = fluid.dygraph.Pool2D(pool_size=3, pool_type='max', pool_stride=1, pool_padding=1)
        # (x-3+2)/1 + 1= x
        self.conv05_1 = ConvBNLayer(num_channels=512, num_filters=512, filter_size=3, padding=1, act="relu")
        self.conv05_2 = ConvBNLayer(num_channels=512, num_filters=512, filter_size=3, padding=1, act="relu")
        self.conv05_3 = ConvBNLayer(num_channels=512, num_filters=512, filter_size=3, padding=1, act="relu")
        # n*512*80*60

        # shallow network
        self.conv_s_1 = ConvBNLayer(num_channels=3, num_filters=24, filter_size=5, padding=2, stride=1, act='relu')
        self.pool_s_1 = fluid.dygraph.Pool2D(pool_size=2, pool_type='avg', pool_stride=2, pool_padding=0)

        self.conv_s_2 = ConvBNLayer(num_channels=24, num_filters=24, filter_size=5, padding=2, stride=1, act='relu')
        self.pool_s_2 = fluid.dygraph.Pool2D(pool_size=2, pool_type='avg', pool_stride=2, pool_padding=0)

        self.conv_s_3 = ConvBNLayer(num_channels=24, num_filters=24, filter_size=5, padding=2, stride=1, act='relu')
        self.pool_s_3 = fluid.dygraph.Pool2D(pool_size=2, pool_type='avg', pool_stride=2, pool_padding=0)
        # n*24*80*60

        # 通道维连接
        # input:512+24 *80*60
        self.connect_conv = ConvBNLayer(num_channels=512 + 24, num_filters=1, filter_size=1, padding=0, stride=1)
        # output:1*80*60
        # 上采样,双线性插值生成label密度图的大小的图

        # 最后使用L2 loss计算损失

    def forward(self, inputs, label=None):
        """前向计算"""
        # deep net
        out = self.conv01_1(inputs)
        out = self.conv01_2(out)
        out = self.pool01(out)

        out = self.conv02_1(out)
        out = self.conv02_2(out)
        out = self.pool02(out)

        out = self.conv03_1(out)
        out = self.conv03_2(out)
        out = self.conv03_3(out)
        out = self.pool03(out)

        out = self.conv04_1(out)
        out = self.conv04_2(out)
        out = self.conv04_3(out)
        out = self.pool04(out)
        out = self.conv05_1(out)
        out = self.conv05_2(out)
        out = self.conv05_3(out)

        # shadow net
        out1 = self.conv_s_1(inputs)
        out1 = self.pool_s_1(out1)

        out1 = self.conv_s_2(out1)
        out1 = self.pool_s_2(out1)

        out1 = self.conv_s_3(out1)
        out1 = self.pool_s_3(out1)

        # concatenate
        # 按通道维连接
        conn = fluid.layers.concat(input=[out, out1], axis=1)
        result = self.connect_conv(conn)
        return result
class CNN(fluid.dygraph.Layer):
    '''
    网络
    '''

    def __init__(self):
        super(CNN, self).__init__()

        self.conv01_1 = fluid.dygraph.Conv2D(num_channels=3, num_filters=64, filter_size=3, padding=1, act="relu")
        self.pool01 = fluid.dygraph.Pool2D(pool_size=2, pool_type='max', pool_stride=2)

        self.conv02_1 = fluid.dygraph.Conv2D(num_channels=64, num_filters=128, filter_size=3, padding=1, act="relu")
        self.pool02 = fluid.dygraph.Pool2D(pool_size=2, pool_type='max', pool_stride=2)

        self.conv03_1 = fluid.dygraph.Conv2D(num_channels=128, num_filters=256, filter_size=3, padding=1, act="relu")
        self.pool03 = fluid.dygraph.Pool2D(pool_size=2, pool_type='max', pool_stride=2)

        self.conv04_1 = fluid.dygraph.Conv2D(num_channels=256, num_filters=512, filter_size=3, padding=1, act="relu")

        self.conv05_1 = fluid.dygraph.Conv2D(num_channels=512, num_filters=512, filter_size=3, padding=1, act="relu")

        self.conv06 = fluid.dygraph.Conv2D(num_channels=512, num_filters=256, filter_size=3, padding=1, act='relu')
        self.conv07 = fluid.dygraph.Conv2D(num_channels=256, num_filters=128, filter_size=3, padding=1, act='relu')
        self.conv08 = fluid.dygraph.Conv2D(num_channels=128, num_filters=64, filter_size=3, padding=1, act='relu')
        self.conv09 = fluid.dygraph.Conv2D(num_channels=64, num_filters=1, filter_size=1, padding=0, act=None)

    def forward(self, inputs, label=None):
        """前向计算"""
        out = self.conv01_1(inputs)

        out = self.pool01(out)

        out = self.conv02_1(out)

        out = self.pool02(out)

        out = self.conv03_1(out)

        out = self.pool03(out)

        out = self.conv04_1(out)

        out = self.conv05_1(out)
        out = self.conv06(out)
        out = self.conv07(out)
        out = self.conv08(out)
        out = self.conv09(out)

        return out


#定义不同返回类型
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
#返回文件类型
def allowed_file_vedio(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS_VEDIO
#返回mp4


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
'''
#给视频创建相应文件夹
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
    else:
        print
        "---  There is this folder!  ---"
'''
@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的文件类型，仅限于png、PNG、jpg、JPG、bmp、mp4"})



        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        # upload_path = os.path.join(basepath, 'static/images','test.jpg')  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)



        #如果上传的是视频，则将视频转为图片，并计算每一帧中人数总和
        if (f and allowed_file_vedio(f.filename)):

            #视频转图片
            vc = cv2.VideoCapture(upload_path)  # 读入视频文件，命名vc

            if vc.isOpened():  # 判断是否正常打开
                rval, frame = vc.read()
            else:
                rval = False

            timeF = 200  # 视频帧计数间隔频率

            new_dir = './static/images/' + f.filename + '_pic'

            if not os.path.exists(new_dir):
                os.mkdir(new_dir)

            i = 0  #计数选取了多少张图片
            n = 1  #初始化为0
            while rval:  # 循环读取视频帧
                rval, frame = vc.read()
                if (n % timeF == 0):  # 每隔timeF帧进行存储操作
                    i += 1
                    cv2.imwrite(new_dir + '/{}.jpg'.format(i), frame)  # 存储为图像
                    print(i,n)
                n = n + 1
                cv2.waitKey(1)
            vc.release()

            #i = 104
            people = 0#记录总人数
            people_vedio = {}
            #对得到的图片进行处理
            for a in range(i):
                if a == 0:
                    continue
                with fluid.dygraph.guard():
                    model, _ = fluid.dygraph.load_dygraph("cnn")
                    cnn = CNN()
                    cnn.load_dict(model)
                    cnn.eval()

                    test_img = Image.open(new_dir + '/{}.jpg'.format(a))
                    test_img = test_img.resize((640, 480))
                    test_im = np.array(test_img)
                    test_im = test_im / 255.0
                    test_im = test_im.transpose().reshape(3, 640, 480).astype('float32')
                    dy_x_data = np.array(test_im).astype('float32')
                    dy_x_data = dy_x_data[np.newaxis, :, :, :]
                    nimg = fluid.dygraph.to_variable(dy_x_data)
                    out = cnn(nimg)
                    temp = out[0][0]
                    temp = temp.numpy()
                    people_vedio[a] = int(np.sum(temp))

                    people = people + people_vedio[a]
                    people_new = int(people / i)

            time.sleep(10)
            return render_template('upload_ok.html', output=people_new, val1=time.time())

        #如果上传的不是视频而是图片，则对单张图片进行计数
        else:
            # 使用Opencv转换一下图片格式和名称
            img = cv2.imread(upload_path)
            cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)

            with fluid.dygraph.guard():
                model, _ = fluid.dygraph.load_dygraph("cnn")
                cnn = CNN()
                cnn.load_dict(model)
                cnn.eval()

                test_img = Image.open(upload_path)
                test_img = test_img.resize((640, 480))
                test_im = np.array(test_img)
                test_im = test_im / 255.0
                test_im = test_im.transpose().reshape(3, 640, 480).astype('float32')
                dy_x_data = np.array(test_im).astype('float32')
                dy_x_data = dy_x_data[np.newaxis, :, :, :]
                nimg = fluid.dygraph.to_variable(dy_x_data)
                out = cnn(nimg)
                temp = out[0][0]
                temp = temp.numpy()
                people = int(np.sum(temp))
                print(people)
            time.sleep(1)
            return render_template('upload_ok.html',output=people, val1=time.time())

    return render_template('upload.html')





'''
@app.route('/hello', methods=['POST', 'GET'])  # 添加路由
def hello():

        # 以POST方式传参数，通过form取值
        # 如果Key之不存在，报错KeyError，返回400的页面
    myout = densehere.test()

    return render_template(output=myout)
'''
from wsgiref.simple_server import make_server
'''
if __name__ == '__main__':
    # app.debug = True
    server=make_server('127.0.0.1',5000,app)
    server.serve_forever()
    app.run()

app.run(port=5000)
'''


#段文思-7.14-开始
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite://'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.secret_key = '\xc9ixnRb\xe40\xd4\xa5\x7f\x03\xd0y6\x01\x1f\x96\xeao+\x8a\x9f\xe4'
db = SQLAlchemy(app)

####################################
#数据库
####################################

#定义ORM
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True)
    password = db.Column(db.String(80))
    email = db.Column(db.String(120), unique=True)

    def __repr__(self):
        return '<User %r>' % self.username


# 创建表格、插入数据
@app.before_first_request
def create_db():
    db.drop_all()  # 每次运行，先删除再创建
    db.create_all()

    admin = User(username='admin', password='root', email='admin@example.com')
    db.session.add(admin)

    guestes = [User(username='guest1', password='guest1', email='guest1@example.com'),
               User(username='guest2', password='guest2', email='guest2@example.com'),
               User(username='guest3', password='guest3', email='guest3@example.com'),
               User(username='guest4', password='guest4', email='guest4@example.com')
               ]
    db.session.add_all(guestes)
    db.session.commit()


############################################
# 辅助函数、装饰器
############################################

# 登录检验（用户名、密码验证）
def valid_login(username, password):
    user = User.query.filter(and_(User.username == username, User.password == password)).first()
    if user:
        return True
    else:
        return False


# 注册检验（用户名、邮箱验证）
def valid_regist(username, email):
    user = User.query.filter(or_(User.username == username, User.email == email)).first()
    if user:
        return False
    else:
        return True

# 登录
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # if g.user:
        if session.get('username'):
            return func(*args, **kwargs)
        else:
            return redirect(url_for('sign_in', next=request.url))
    return wrapper

############################################
# 路由
############################################
#1.登录
@app.route('/sign_in',methods=['POST', 'GET'])
def sign_in():
    error = None
    if request.method == 'POST':
        if valid_login(request.form['username'], request.form['password']):
            flash("成功登录！")
            session['username'] = request.form.get('username')
            return render_template('upload.html')
        else:
            error = '错误的用户名或密码！'
    return render_template('sign_in.html', error = error)

#2.注册
@app.route('/sign_up', methods=['POST', 'GET'])
def sign_up():
    error = None
    if request.method == 'POST':
        if valid_regist(request.form['username'], request.form['email']):
            user = User(username=request.form['username'], password=request.form['password1'],
                        email=request.form['email'])
            db.session.add(user)
            db.session.commit()

            flash("成功注册！")
            return redirect(url_for('sign_in'))
        else:
            error = '该用户名或邮箱已被注册！'
    return render_template('sign_up.html', error = error)
#段文思-7.14-结束


if __name__=="__main__":
    app.run( port=5000)
