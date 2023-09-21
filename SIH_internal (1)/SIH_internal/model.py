from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    username = db.Column(db.String,primary_key = True,unique = True,nullable = False)
    email = db.Column(db.String,nullable = False,unique = True)
    password = db.Column(db.String,nullable = False)


