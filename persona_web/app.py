from flask import Flask, render_template, request

app = Flask(__name__)



#model -> text 

# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
@app.route('/', methods = ['GET', 'POST'])
def index():

	# client -> server 
	if request.method == 'POST':
		print('request: ', request.data)

	elif request.method == 'GET':
		print('get method')

	# server -> client
	return render_template('index.html', text = 'asd')

if __name__ == '__main__':

	app.run(host = '0.0.0.0', debug=True)
