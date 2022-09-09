from app import app
print(__name__)
if __name__=='__main__':
    app.run()
    # app.run(debug=True, host='0.0.0.0', port=5000, threaded=False)