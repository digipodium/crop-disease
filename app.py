import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageEnhance
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import base64

app = Flask(__name__)
app.config['SECRET_KEY'] = 'agriscan-ai-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- MODELS ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    disease_name = db.Column(db.String(200), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    image_filename = db.Column(db.String(300), nullable=False)
    crop_type = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- AI CORE CONFIG ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_names = [
    'Pepper__bell___Bacterial_spot', 
    'Pepper__bell___healthy', 
    'Unknown / Not a disease', 
    'Potato___Early_blight', 
    'Potato___Late_blight', 
    'Potato___healthy', 
    'Tomato_Bacterial_spot', 
    'Tomato_Early_blight', 
    'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 
    'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 
    'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 
    'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'
]

# Robust TenCrop preprocessing
robust_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(224),
    transforms.Lambda(lambda crops: torch.stack([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(transforms.ToTensor()(crop)) 
        for crop in crops
    ]))
])

def load_disease_model():
    model_path = os.path.join(os.path.dirname(__file__), 'crop_disease_model.pth')
    model = models.mobilenet_v3_large(weights=None, num_classes=len(class_names))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model file {model_path} not found.")
    model.to(DEVICE)
    model.eval()
    return model

MODEL = load_disease_model()

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        
        new_user = User(name=name, email=email, password=generate_password_hash(password, method='pbkdf2:sha256'))
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# --- SCANNING CORE ---
def get_model_prediction(image):
    # Enhance slightly
    image = ImageEnhance.Contrast(image).enhance(1.1)
    image = ImageEnhance.Sharpness(image).enhance(1.2)
    
    # Process batch (10 views)
    input_batch = robust_preprocess(image).to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(input_batch)
        avg_output = outputs.mean(0)
        probabilities = torch.nn.functional.softmax(avg_output, dim=0)
        top_probs, top_indices = torch.topk(probabilities, 2)
        
    res_index = top_indices[0].item()
    conf = top_probs[0].item() * 100
    
    # Unknown logic
    if res_index == 2 and top_probs[1].item() > 0.10:
        res_index = top_indices[1].item()
        conf = top_probs[1].item() * 100
        
    return class_names[res_index], conf

@app.route('/demo_predict/<path:filepath>')
def demo_predict(filepath):
    safe_path = os.path.normpath(filepath).replace('..', '')
    full_path = os.path.join(os.getcwd(),'PlantVillage', safe_path)
    
    if os.path.exists(full_path):
        image = Image.open(full_path).convert('RGB')
        result, conf = get_model_prediction(image)
        
        # Prepare for template
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        return render_template('results.html', result=result, confidence=f"{conf:.2f}", image_data=encoded_image)
    return redirect(url_for('upload'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(url_for('upload'))
        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('upload'))
        
        if file:
            try:
                img_bytes = file.read()
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                
                result, conf = get_model_prediction(image)
                
                # Save to history
                filename = secure_filename(f"{current_user.id}_{int(datetime.now().timestamp())}_{file.filename}")
                image_save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.seek(0)
                file.save(image_save_path)
                
                crop_type = "Unknown"
                if "Tomato" in result: crop_type = "Tomato"
                elif "Potato" in result: crop_type = "Potato"
                elif "Pepper" in result: crop_type = "Pepper"

                new_pred = Prediction(
                    user_id=current_user.id,
                    disease_name=result,
                    confidence=conf,
                    image_filename=filename,
                    crop_type=crop_type
                )
                db.session.add(new_pred)
                db.session.commit()
                
                encoded_image = base64.b64encode(img_bytes).decode('utf-8')
                return render_template('results.html', result=result, confidence=f"{conf:.2f}", image_data=encoded_image)
            except Exception as e:
                flash(f"Error: {str(e)}", "error")
                return redirect(url_for('upload'))
                
    # GET: Scan page logic
    demos = []
    pv_path = 'PlantVillage'
    if os.path.exists(pv_path):
        for d in os.listdir(pv_path):
            d_path = os.path.join(pv_path, d)
            if os.path.isdir(d_path) and d != 'PlantVillage':
                imgs = [f for f in os.listdir(d_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if imgs:
                    demos.append({'class': d, 'path': os.path.join(d, imgs[0])})
    
    return render_template('upload.html', demo_images=demos[:4])

@app.route('/history')
@login_required
def history():
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
    return render_template('history.html', predictions=user_predictions)

@app.route('/delete_history/<int:pred_id>')
@login_required
def delete_history(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id == current_user.id:
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], pred.image_filename))
        except:
            pass
        db.session.delete(pred)
        db.session.commit()
        flash("Record deleted", "success")
    return redirect(url_for('history'))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)