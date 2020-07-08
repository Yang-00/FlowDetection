from flask import Flask,request,render_template,redirect

app = Flask(__name__)

@app.route("/",methods=['GET','POST'])
def login():
    if request.method =='POST':
        username = request.form['username']
        password = request.form['password']
        if username =="user" and password=="password":
            return redirect("https://www.whu.edu.cn/")
        else:
            message = "Failed Login"
            return render_template('login.html',message=message)
    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)