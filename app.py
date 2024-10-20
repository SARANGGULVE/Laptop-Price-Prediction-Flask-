from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app and specify template folder if necessary
app = Flask(__name__, template_folder='templates')

# Load the trained model and dataframe for input reference
pipe = pickle.load(open('D:\\Data Science\\project\\Laptop_Price_Prediction\\pipe.pkl', 'rb'))
df = pickle.load(open('D:\\Data Science\\project\\Laptop_Price_Prediction\\df.pkl', 'rb'))

@app.route('/', methods=["GET"])
def index():
    # Extract unique options for dropdown fields from the dataframe
    companies = sorted(df['Company'].unique())
    types = sorted(df['TypeName'].unique())
    ram = sorted(df['Ram'].unique())
    weight = sorted(df['Weight'].unique())
    touchscreen = ['No', 'Yes']
    ips = ['No', 'Yes']
    resolutions = ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
                   '2880x1800', '2560x1600', '2560x1440', '2304x1440']
    cpu_brands = sorted(df['Cpu brand'].unique())
    hdd = [0, 128, 256, 512, 1024, 2048]
    gpu_brands = sorted(df['Gpu brand'].unique())
    os = sorted(df['os'].unique())

    # Render the template with form options
    return render_template('index.html', companies=companies, types=types, ram=ram,
                           weight=weight, touchscreen=touchscreen, ips=ips,
                           resolutions=resolutions, cpu_brands=cpu_brands, 
                           gpu_brands=gpu_brands, os=os)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve values from the form
    company = request.form.get('company')
    typename = request.form.get('typename')
    ram = int(request.form.get('ram'))
    weight = float(request.form.get('weight'))
    touchscreen = 1 if request.form.get('touchscreen') == 'Yes' else 0
    ips = 1 if request.form.get('ips') == 'Yes' else 0
    resolution = request.form.get('resolution')
    screen_size = float(request.form.get('screen_size'))  # Diagonal size in inches
    cpu_brand = request.form.get('cpu_brand')
    hdd = request.form.get('hdd')
    ssd = int(request.form.get('ssd'))
    gpu_brand = request.form.get('gpu_brand')
    os = request.form.get('os')

    # Calculate PPI based on resolution and screen size
    x_res, y_res = map(int, resolution.split('x'))
    ppi = ((x_res**2 + y_res**2) ** 0.5) / screen_size

    # Create input array for the model
    query = np.array([company, typename, ram, weight, touchscreen, ips, ppi, cpu_brand, hdd, ssd, gpu_brand, os])

    # Predict the price
    prediction = np.exp(pipe.predict([query])[0])

    # Render the same index.html template with the prediction result
    return render_template('index.html', 
                           prediction_text=f"The predicted price of the laptop is: {int(prediction)}", 
                           companies=sorted(df['Company'].unique()),
                           types=sorted(df['TypeName'].unique()),
                           ram=sorted(df['Ram'].unique()),
                           weight=sorted(df['Weight'].unique()),
                           touchscreen=['No', 'Yes'],
                           ips=['No', 'Yes'],
                           resolutions=['1920x1080', '1366x768', '1600x900', 
                                        '3840x2160', '3200x1800', '2880x1800', 
                                        '2560x1600', '2560x1440', '2304x1440'],
                           cpu_brands=sorted(df['Cpu brand'].unique()),
                           hdd = [0, 128, 256, 512, 1024, 2048],
                           gpu_brands=sorted(df['Gpu brand'].unique()),
                           os=sorted(df['os'].unique()))

if __name__ == '__main__':
    app.run(debug=True)
