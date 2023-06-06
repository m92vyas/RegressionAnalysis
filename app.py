from flask import Flask,request,render_template
from src.pipeline import predict_pipeline

app=Flask(__name__,template_folder='templates')

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':

        return render_template('home.html')
    
    else:
        data=predict_pipeline.features_to_numpy(
            quantity_tons=float(request.form.get('quantity_tons')),
            status=request.form.get('status'),
            application=float(request.form.get('application')),
            thickness=float(request.form.get('thickness')),
            width=float(request.form.get('width')),
            product_ref=float(request.form.get('product_ref')),
            delivery_date=float(request.form.get('delivery_date'))

        )

        prediction=predict_pipeline.PredictPipeline()
        results=prediction.predict(data)
        return render_template('home.html',results=results)
    