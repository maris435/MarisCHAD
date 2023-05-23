import config
from flask import Flask, request, render_template
from flask_wtf import CSRFProtect
from flask_sqlalchemy import SQLAlchemy

from datetime import date

from chatbot import chatbot

# Initialize CSRF protection, app and its components
csrf = CSRFProtect()


app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = config.SQLALCHEMY_DATABASE_URI
app.config['SECRET_KEY'] = config.APP_SECRET_KEY
db = SQLAlchemy(app)
csrf.init_app(app)

# Initialize the tables to be added to a database
class Session(db.Model):
    session_id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date)

class Conversation(db.Model):
    question_id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('session.session_id'))
    question = db.Column(db.String(1024))
    answer = db.Column(db.String(1024))

# Routing the homepage
@app.route("/", methods=["GET", "POST"])
def ask_question():

    # Create an instance of the chatbot
    my_chatbot = chatbot()

    # Check if the request is a POST request
    if request.method == 'POST':

        # Get the question from the form user
        question = request.form.get('question')

        if question:
            # Get the response from the chatbot
            response = my_chatbot.run(input=question)

            # Get the latest session
            session = Session.query.order_by(Session.session_id.desc()).first()

            # Check if a new session needs to be created- if id does, add to database
            if session is None or session.date != date.today():
                session = Session(date=date.today())
                db.session.add(session)
                db.session.commit()

            # Create a new conversation entry with the session, question, and response
            conversation = Conversation(session_id=session.session_id, question=question, answer=response)

            # Add the conversation to the database session, commit changes
            db.session.add(conversation)
            db.session.commit()

            # Render the template with the response
            return render_template("index.html", answer=response)

    # Render the template with no response
    return render_template("index.html", answer=None)

# Routing the privacy policy page
@app.route('/privacy_policy')
def privacy_policy():
    # Render the template with privacy policy
    return render_template('privacy_policy.html')


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        app.run(debug=True, port=5001)
