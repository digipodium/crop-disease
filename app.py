from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import re
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

app = Flask(__name__)
app.secret_key = 'secretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Database and login manager setup
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure upload folder exists before handling requests
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# User / Prediction models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    disease_name = db.Column(db.String(200), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    image_filename = db.Column(db.String(300), nullable=False)
    crop_type = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)



# Model and Class Names will be initialized below to avoid redundancy.

# Prediction helpers

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def format_label(label):
    return re.sub(r'_+', ' ', label).strip()


from PIL import ImageEnhance

def preprocess_image(image_file):
    image = Image.open(image_file).convert('RGB')
    
    # Enhance Image for better feature extraction (helps with Google images)
    image = ImageEnhance.Contrast(image).enhance(1.2)
    image = ImageEnhance.Sharpness(image).enhance(1.5)
    
    preprocess_pipeline = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return preprocess_pipeline(image).unsqueeze(0).to(device)


def predict_disease(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, idx = torch.max(probabilities, dim=1)
        label = class_names[idx.item()]
        return format_label(label), float(confidence.item() * 100)


# ---------------- MODEL LOADING ----------------
with app.app_context():
    db.create_all()
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 16 # Adjusted to 16 to match trained weights

# Load model architecture
model = models.mobilenet_v3_large(weights=None)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, num_classes)

# Load saved weights
MODEL_PATH = "crop_disease_model.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
else:
    print(f"Warning: Model file {MODEL_PATH} not found.")

# Optimized preprocessing pipeline (Used for global consistency)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class_names = [
    'Pepper__bell___Bacterial_spot', 
    'Pepper__bell___healthy', 
    'PlantVillage', 
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


# -------- PASSWORD VALIDATION FUNCTION --------
def valid_password(password):

    if len(password) < 8:
        return False

    if " " in password:
        return False

    if not re.search("[A-Z]", password):   # uppercase
        return False

    if not re.search("[a-z]", password):   # lowercase
        return False

    if not re.search("[0-9]", password):   # number
        return False

    if not re.search("[@#$%^&*!]", password):  # special character
        return False

    return True


# -------- REGISTER --------
@app.route('/register', methods=['GET','POST'])

def register():

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # validations 
        if not name or len(name.strip())<2:
            flash('Name must be at least 2 characters long.', 'error')
            return redirect(url_for('register'))

        if not re.match(r'^[\w\.-]+@gmail\.com$', email):
            flash('Enter valid Gmail (example@gmail.com)', 'error')
            return redirect(url_for('register'))
        
        if not valid_password(password):
            flash('Password must contain uppercase, lowercase, number and special character', 'error')
            return redirect(url_for('register'))
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        #check if user already exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered.Please log in.', 'error')
            return redirect(url_for('register'))

        # create new user
        hashed_password = generate_password_hash(password)
        new_user = User(name=name.strip(), 
                        email=email.strip(),
                        password=hashed_password
        )
        try: 
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            # after sign-up send user to login page
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'error')
            return redirect(url_for('register'))

    return render_template('register.html')


# -------- LOGIN --------
@app.route('/login', methods=['GET','POST'])

def login():

    if request.method == 'POST':


        email = request.form['email'].strip()
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password,password):
            login_user(user)
            session['user_id'] = user.id
            session['user_name'] = user.name

            flash("Login successful","success")
            return redirect(url_for('index'))

        else:
            flash("Invalid email or password","error")

    return render_template('login.html')




# -------- LOGOUT --------
@app.route('/logout')
def logout():
    logout_user()
    session.clear()
    flash("Logged out successfully","success")

    return redirect(url_for('login'))


# -------- ABOUT --------
@app.route('/about')
def about():
    return render_template('about.html',current_user =current_user )


# -------- INDEX --------
@app.route('/')
def index():
    return render_template('index.html', current_user=current_user)

@app.route('/plantvillage/<path:filename>')
def serve_plantvillage(filename):
    return send_from_directory('PlantVillage', filename)

@app.route('/demo_predict/<path:filepath>')
@login_required
def demo_predict(filepath):
    # Protect against path traversal
    safe_path = os.path.normpath(filepath).replace('..', '')
    full_path = os.path.join('PlantVillage', safe_path)
    
    if not os.path.exists(full_path):
        flash("Sample image not found", "error")
        return redirect(url_for('upload'))
    
    try:
        with open(full_path, 'rb') as f:
            img_bytes = f.read()
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            # Preprocess
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                output = model(input_batch)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, index = torch.max(probabilities, 0)
            
            # Match the class name (index adjustment handled in app initialization)
            result = class_names[index.item()]
            conf_score = confidence.item() * 100
            
            import base64
            encoded_image = base64.b64encode(img_bytes).decode('utf-8')
            
            return render_template('results.html', 
                                 result=result, 
                                 confidence=f"{conf_score:.2f}",
                                 image_data=encoded_image)
    except Exception as e:
        flash(f"Error processing demo image: {str(e)}", "error")
        return redirect(url_for('upload'))

def get_demo_images():
    root = 'PlantVillage'
    demos = []
    try:
        # Get one image per class
        classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        for cls in classes:
            cls_path = os.path.join(root, cls)
            files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                # Store full path for predict and local path for display
                demos.append({
                    'class': cls,
                    'file': files[0],
                    'path': f"{cls}/{files[0]}"
                })
    except:
        pass
    return demos

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part", "error")
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash("No selected file", "error")
            return redirect(request.url)
        
        if file:
            try:
                # Read image
                img_bytes = file.read()
                # Preprocess with Enhancement
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                image = ImageEnhance.Contrast(image).enhance(1.2)
                image = ImageEnhance.Sharpness(image).enhance(1.5)
                
                input_tensor = preprocess(image)
                input_batch = input_tensor.unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    output = model(input_batch)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    confidence, index = torch.max(probabilities, 0)
                
                result = class_names[index.item()]
                conf_score = confidence.item() * 100
                
                # Convert image to base64 for display
                import base64
                encoded_image = base64.b64encode(img_bytes).decode('utf-8')
                
                return render_template('results.html', 
                                     result=result, 
                                     confidence=f"{conf_score:.2f}",
                                     image_data=encoded_image)
            except Exception as e:
                flash(f"Error processing image: {str(e)}", "error")
                return redirect(request.url)

    demo_images = get_demo_images()
    return render_template('upload.html', demo_images=demo_images)

if __name__ == "__main__":
    app.run(debug=True)