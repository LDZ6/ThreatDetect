import os

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from Webshell_Detect_FC import detect_Webshell
from PE_Detect_FC import PE_detect

app = Flask(__name__)

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'php', 'jsp', 'asp', 'exe', 'dll'}
WEBSHELL_EXTENSIONS = {'php', 'jsp', 'asp'}
PE_EXTENSIONS = {'exe', 'dll'}

@app.route('/', methods=['GET', 'POST'])
def index():
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    if request.method == 'POST':
        # 检查是否有文件部分
        if 'file' not in request.files:
            return jsonify(message='No file part')
        file = request.files['file']
        if file.filename == '':
            return jsonify(message='No selected file')
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
            if allowed_file(file.filename, WEBSHELL_EXTENSIONS):
                filename = secure_filename(file.filename)
                dir = 'uploads'
                filepath = os.path.join(dir, filename)
                file.save(filepath)
                result, MD5 = detect_Webshell(filepath)
                if result == 0:
                    result = "正常文件"
                elif result == 0.5:
                    result = '可疑文件'
                elif result == 1:
                    result = "Webshell"
                # 返回JSON对象
                # clear_directory(dir)
                file_id = request.form.get('file_id')  # 获取传递的 id
                return jsonify(result=result, MD5=MD5, id=file_id)
            elif allowed_file(file.filename, PE_EXTENSIONS):
                filename = secure_filename(file.filename)
                dir = 'uploads'
                filepath = os.path.join(dir, filename)
                file.save(filepath)
                result, MD5 = PE_detect(filepath)
                if result == 0:
                    result = "正常软件"
                elif result == 0.5:
                    result = '可疑软件'
                elif result == 1:
                    result = "恶意软件"
                # 返回JSON对象
                file_id = request.form.get('file_id')  # 获取传递的 id
                return jsonify(result=result, MD5=MD5, id=file_id)

    # 如果是GET请求，返回上传页面
    return render_template('selfpage.html')

def allowed_file(filename, ALLOWED_EXTENSIONS):
    # 检查文件扩展名是否允许
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(debug=True)
