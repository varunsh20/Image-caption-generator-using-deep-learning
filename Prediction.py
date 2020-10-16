import PIL
from flask import Flask, render_template, request

import Generate_caption

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/',methods=["POST"])
def prediction():
    if request.method=="POST":
        f = request.files['userfile']
        path = 'static/{}'.format(f.filename)
        f.save(path)
        caption = Generate_caption.caption_image(path)
        result = {'image':path,'cap':caption}
    return render_template('index.html',your_caption = result)
if __name__== '__main__':
    app.run(debug=True)

