from distutils.log import debug
import re
from flask import Flask, render_template, request
import fm22
app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def hello():
    if request.method == "POST":
        playernamee = request.form['playername']
        wage = fm22.fm22_prediction(playernamee)
        print(wage)
    else:
        wage = 0
    return render_template("index.html", wage_s=wage)
"""
@app.route("/sub", methods = ['POST'])
def submit():
    if request.method == "POST":
        name = request.form["username"]
    
    return render_template("sub.html", n=name)
"""
if __name__ == "__main__":
    app.run(debug=True)