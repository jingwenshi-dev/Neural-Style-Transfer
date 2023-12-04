from flask import Flask, render_template, request, send_from_directory
import NST
import os
from werkzeug.utils import secure_filename

from torchvision.utils import save_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'outputs/'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/transform', methods=['POST'])
def transform():
    if request.method == 'POST':
        content_file = request.files['content_image']
        style_file = request.files['style_image']

        if content_file and style_file:
            content_filename = secure_filename(content_file.filename)
            style_filename = secure_filename(style_file.filename)
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
            content_file.save(content_path)
            style_file.save(style_path)

            # Perform Neural Style Transfer
            style_image, content_image, input_image = NST.image_loader(style_path, content_path)
            output = NST.start_NST(optimizer="lbfgs",
                                   content_img=content_image,
                                   style_img=style_image,
                                   input_img=input_image,
                                   num_steps=300,
                                   style_weight=1000000)

            output_image_path = os.path.join(app.config['OUTPUT_FOLDER'],
                                             'output_' + content_filename)
            save_image(output, output_image_path)
            return {'image_path': output_image_path}


if __name__ == '__main__':
    app.run(debug=True)
