from flask import Flask,render_template,redirect,request, url_for
from model import *
import numpy as np
import pandas as pd
import pickle
import numpy as np





s_dict={'Business': 0, 'Retired': 1, 'Salaried': 2, 'Student': 3}
l_dict={'Average': 0, 'Luxury': 1, 'Minimalist': 2}
g_dict={'No': 0, 'Yes': 1}
d_dict={'Daily': 0, 'Never': 1, 'Rarely': 2, 'Weekly': 3}
w_dict={'Daily': 0, 'Never': 1, 'Rarely': 2, 'Weekly': 3}
sw_dict={'No': 0, 'Yes': 1}
st_dict={'Direct Supply': 0, 'Tank': 1, 'Well': 2}

cal_model_path = 'rf_pipeline.pkl'
calculate = pickle.load(open(cal_model_path,'rb'))



#==============================configuration===============================

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///database.sqlite3"
db.init_app(app)
app.app_context().push()

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('landing.html')

# #==============================Login=================================
# @app.route('/login',methods=['GET','POST'])
# def login():
#     if request.method == "GET":
#         return render_template("login.html")
#     elif request.method == "POST":
#         a = request.form['username']
#         b = request.form['password']
#         y = User.query.all()
#         l = [] 
#         for i in y:
#            l.append((i.username,i.password))
#         if (a,b) in l:
#             return redirect(url_for('user_dashboard',id = a))
#         else:
#             return redirect('/unsuccessful')


# #==============================Signup=================================
# @app.route('/signup',methods=['GET','POST'])
# def signup():
#     if request.method == 'GET':
#         return render_template('signup.html')
#     if request.method == 'POST':
#         a = request.form['username']
#         b = request.form['password']
#         c = request.form['email']
#         user_record = User(username=a, email=c, password=b)
#         y = User.query.all()
#         l = []
#         for i in y:
#             l.append(i.username)
#         if a in l:
#             return redirect('/success')
#         else:
#             db.session.add(user_record)
#             db.session.commit()
#             return redirect('/success')

# #==============================User Dashboard=================================
# @app.route('/user_dashboard/<id>',methods=['GET','POST'])
# def user_dashboard(id):
#     return render_template('landing.html',name=id)

@app.route('/calculates',methods=['GET','POST'])
def calculates():

    if request.method == 'GET':
        return render_template('calculate.html')
    if request.method == 'POST':
        a = request.form['Number_of_people']
        b = request.form['Average_Age']
        c = request.form['Total_Income']
        d = request.form['Occupation']
        f = request.form['Lifestyle']
        g = request.form['House_Size']
        h = request.form['Garden']
        i = request.form['Number_of_bathrooms']
        j = request.form['Dishwasher_usage']
        k = request.form['Washing_machine_usage']
        l = request.form['Swimming']
        m = request.form['Water_storage']

        l1 = [a,b,c,s_dict[d],l_dict[f],g,g_dict[h],i,d_dict[j],w_dict[k],sw_dict[l],st_dict[m]]
        data1 = [np.array(l1)]
        print(data1)
        my_prediction = calculate.predict(data1)
        final_prediction = my_prediction[0]
        print('pred',my_prediction[0])
        # final_prediction = label_dict[final_prediction]

        return render_template('calculate.html', prediction_text=final_prediction)




@app.route('/success',methods=['GET','POST'])
def success():
    return render_template('success.html')
if __name__ == "__main__":
    app.run(debug=True)