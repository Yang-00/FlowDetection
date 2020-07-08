# 在学习关于flask的知识点，这是在csdn上面找的一篇文章，主要内容关于图片的上传
'''
版权声明：本文为CSDN博主「Java知音_」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_36380516/java/article/details/80347192
'''

from flask import Flask, render_template, request
import os

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))


@app.route("/up_photo", methods=['post', 'get'])
def up_photo():
    img = request.files.get('txt_photo')
    username = request.form.get("name")
    path = basedir + "/static/photo/"
    file_path = path + img.filename
    img.save(file_path)
    print('上传头像成功，上传的用户是：' + username)
    return render_template("图片上传系统.html")

if __name__ == '__main__':
    app.run(debug=False)
