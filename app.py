from flask import Flask, request, jsonify
from ResumeNLPProccesor.py import ResumeNLPProcessor  # replace with actual import

app = Flask(__name__)
processor = ResumeNLPProcessor()

@app.route('/process_resumes', methods=['POST'])
def process_resumes():
    data = request.json
    resumes = data.get('resumes', [])
    if not resumes:
        return jsonify({"error": "No resumes provided"}), 400

    results = processor.process_resume_batch(resumes)
    # We won't return heavy vectors, just metadata
    return jsonify({
        "message": "Resumes processed",
        "num_resumes": len(resumes)
    })


@app.route('/compare', methods=['POST'])
def compare():
    data = request.json
    job_description = data.get('job_description', '')
    resumes = data.get('resumes', [])
    method = data.get('method', 'tfidf')

    if not job_description or not resumes:
        return jsonify({"error": "Missing job_description or resumes"}), 400

    rankings = processor.compare_job_to_resumes(job_description, resumes, method=method)
    # Return ranking with resume index and similarity score
    response = [{"resume_index": idx, "similarity": float(score)} for idx, score in rankings]

    return jsonify({"rankings": response})

if __name__ == '__main__':
    app.run(debug=True)
