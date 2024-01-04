from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ketqua', methods=['POST'])
def tinhtoan():
    txt1 = request.form['txt1']
    txt2 = request.form['txt2']

    # Tính toán độ đo Cosine và Manhattan
    tao_vector = TfidfVectorizer()
    tfidf_matran = tao_vector.fit_transform([txt1, txt2])

    kq_cs = cosine_similarity(tfidf_matran[0:1], tfidf_matran[1:2])
    diem_cosine = kq_cs[0][0]

    kc_mht = manhattan_distances(tfidf_matran[0:1], tfidf_matran[1:2])
    manhattan_diem = kc_mht[0][0]

    # Tìm từ giống nhau và từ riêng của mỗi văn bản
    txt_giongnhau = []
    txt1_khacnhau = []
    txt2_khacnhau = []

    txt_dactrung = tao_vector.get_feature_names_out()
    for tu in txt_dactrung:
        if tfidf_matran[0, tao_vector.vocabulary_[tu]] > 0 and tfidf_matran[1, tao_vector.vocabulary_[tu]] > 0:
            txt_giongnhau.append(tu)
        elif tfidf_matran[0, tao_vector.vocabulary_[tu]] > 0:
            txt1_khacnhau.append(tu)
        elif tfidf_matran[1, tao_vector.vocabulary_[tu]] > 0:
            txt2_khacnhau.append(tu)

    # Gửi kết quả về template
    return render_template('ketqua.html', kq_cs=diem_cosine, kc_mht=manhattan_diem, kq_txt1=txt_giongnhau, kq_txt2=txt1_khacnhau, kq_txt3=txt2_khacnhau)


@app.route('/dem_tu', methods=['POST'])
def dem_tu_xuathien():

    # Lấy từ từ form đã nhập
    tu_kiemtra = request.json['tu']

    # Lấy văn bản về từ js
    txt1 = request.json['txt1']
    txt2 = request.json['txt2']

    dem1 = txt1.count(tu_kiemtra)  # Đếm vb 1
    dem2 = txt2.count(tu_kiemtra)  # Đếm vb 2

    dem_kq = f"Số lần xuất hiện của '{tu_kiemtra}':\nVăn bản 1: {
        dem1}\nVăn bản 2: {dem2}\n"  # Chuẩn bị kq để chuyển

    return jsonify({'dem_kq': dem_kq})  # Gửi kq về json


if __name__ == '__main__':
    app.run(debug=True)
