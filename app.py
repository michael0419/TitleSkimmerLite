import flask
import os
import dill as pickle
import pandas as pd

app = flask.Flask(__name__, template_folder='templates')

path_to_model1_vectorizer = 'models/tf-vectorizer.pkl'
path_to_naive_bayes_model = 'models/mb-text-classifier.pkl'


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))


    if flask.request.method == 'POST':
        # Get the input from the user.
        user_input_text = flask.request.form['user_input_text']
        
        # Turn the text into numbers using our vectorizer
        with open(path_to_model1_vectorizer, 'rb') as f:
            td_idf_vectorizer = pickle.load(f)
        
        # Make a prediction 
        with open(path_to_naive_bayes_model, 'rb') as f:
            nb_model = pickle.load(f)
        
        xIn = td_idf_vectorizer.transform([user_input_text])
        print('Predicted category: ', nb_model.predict(xIn)[0])
        data = {
            'Category': nb_model.classes_,
            'weight': nb_model.predict_proba(xIn)[0],
        }
        results = pd.DataFrame(data)
        results = results.sort_values('weight', ascending=False)
        results['weight'] = results['weight'].apply(lambda x: float('%.3f' % (x*100)))
        results = results.reset_index(drop=True)

        return flask.render_template('index.html', 
            input_text=user_input_text,
            result_nb=results['Category'][0],

            First_nb=results['Category'][0],
            Second_nb=results['Category'][1],
            Third_nb=results['Category'][2],

            percent_first_nb=float('%.3f' %results['weight'][0]),
            percent_second_nb=float('%.3f' %results['weight'][1]),
            percent_third_nb=float('%.3f' %results['weight'][2]),

            mnbayes = "True",
            result = "True",
        )


if __name__ == '__main__':
    app.run(debug=True)