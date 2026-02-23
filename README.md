# 🌊 Remote Sensing Based Water Quality Monitoring using Machine Learning

## 📌 Project Overview
This project develops an intelligent water quality monitoring system using Sentinel-2 satellite imagery and machine learning techniques. The system predicts important inland water quality parameters without manual field testing.

The model estimates:
- Chlorophyll-a
- Dissolved Oxygen (DO)
- Ammonia Nitrogen (NH3-N)

using spectral reflectance data and AI models.

---

## 🎯 Objectives
- Monitor inland water bodies using satellite data
- Reduce dependency on manual water sampling
- Apply machine learning for environmental prediction
- Build a real-time interactive dashboard

---

## 🛰️ Data Source
- Sentinel-2 Satellite Imagery
- Google Earth Engine (GEE)
- Spectral reflectance bands (B2–B8A)

---

## 🧪 Remote Sensing Techniques Used

### NDWI – Normalized Difference Water Index
Used to detect water bodies.

NDWI = (Green - NIR) / (Green + NIR)

Sentinel-2 bands:
- Green → B3
- NIR → B8

---

### MNDWI – Modified NDWI
Improves water detection in urban areas.

MNDWI = (Green - SWIR) / (Green + SWIR)

Bands:
- Green → B3
- SWIR → B11

---

### NDVI – Normalized Difference Vegetation Index
Used to differentiate vegetation from water.

NDVI = (NIR - Red) / (NIR + Red)

Bands:
- NIR → B8
- Red → B4

---

## 🧠 Machine Learning Models Used
- Random Forest Regressor
- Support Vector Regression (SVR)
- XGBoost Regressor
- Artificial Neural Network (ANN)

---

## ⚙️ Methodology
1. User inputs latitude & longitude
2. Sentinel-2 reflectance data retrieved from Google Earth Engine
3. Cloud filtering & preprocessing applied
4. Water detection using NDWI
5. Spectral band extraction (B2–B8A)
6. Model training using ML algorithms
7. Prediction of water quality parameters
8. Dashboard visualization using Streamlit

---

## 🖥️ System Architecture
User Input → GEE Satellite Data → Preprocessing → NDWI Water Detection → Feature Extraction → ML Prediction → Streamlit Dashboard

---

## 📊 Dataset Information
- Dataset size: 806 samples
- Features: Spectral reflectance bands + derived indices
- Train/Test split: 80% / 20%

---

## 📈 Evaluation Metrics
- RMSE (Root Mean Square Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)
- Bias

---

## 🏆 Results Summary

### Chlorophyll-a

| Model | RMSE | R² | MAPE |
|------|------|----|------|
| Random Forest | 1.67 | 0.889 | 4.15 |
| SVR | 3.11 | 0.616 | 6.15 |
| XGBoost | **1.61** | **0.896** | **4.05** |

Best Model → XGBoost

---

### Dissolved Oxygen

| Model | RMSE | R² | MAPE |
|------|------|----|------|
| Random Forest | **2.17** | **0.789** | 21.68 |
| SVR | 3.39 | 0.486 | 26.65 |
| XGBoost | 2.22 | 0.780 | 21.71 |

Best Model → Random Forest

---

### NH3-N

| Model | RMSE | R² | MAPE |
|------|------|----|------|
| Random Forest | 0.69 | 0.952 | 5.48 |
| SVR | 0.99 | 0.901 | 5.52 |
| XGBoost | **0.56** | **0.968** | **4.97** |

Best Model → XGBoost

---

## 💻 Web Application
Built using Streamlit for real-time prediction.

### Features
- Map-based coordinate selection
- Satellite image visualization
- NDWI water detection map
- Real-time ML predictions
- Water quality classification dashboard

---

## 📁 Project Structure

```
water-quality-remote-sensing-ml/
│
├── app.py
├── train_models.py
├── gee_download.py
├── synthetic_data.py
├── utils.py
├── test_water.py
│
├── models/
├── data/
├── results/
│
└── README.md
```

---

## ▶️ Installation & Run

### Step 1: Clone Repository
```
git clone https://github.com/SaiAbhiYaswanth/water-quality-remote-sensing-ml.git
cd water-quality-remote-sensing-ml
```

### Step 2: Install Dependencies
```
pip install -r requirements.txt
```

### Step 3: Run Application
```
streamlit run app.py
```

---

## 📌 Practical Applications
- Environmental monitoring agencies
- Pollution tracking
- Smart city water management
- Inland lake monitoring
- Agricultural water analysis

---

## ⚠️ Limitations
- Dataset limited to specific region
- Satellite cloud interference
- Requires internet for GEE data
- Seasonal variation not fully captured

---

## 🔮 Future Enhancements
- Integration with IoT water sensors
- Drone-based imagery
- Mobile application alerts
- Multi-season training data
- Real-time continuous monitoring

---

## 📚 References
1. Nechad et al., 2010 – Satellite-based turbidity estimation  
2. Dogliotti et al., 2015 – Multi-band turbidity algorithm  
3. Pahlevan et al., 2017 – Sentinel-2 water monitoring  
4. Vanhellemont, 2019 – Atmospheric correction techniques  
5. Dekker et al., 2002 – Remote sensing for inland water  
6. Matthews, 2011 – Multispectral aquatic monitoring  
7. Pahlevan et al., 2020 – Sentinel aquatic algorithms  
8. Chen et al., 2021 – ML-based chlorophyll prediction  
9. Wang et al., 2022 – RF & SVR water quality modeling  
10. Li et al., 2023 – XGBoost environmental prediction  

---

## 👨‍💻 Author
**Aravapalli Sai Abhi Yaswanth**

B.Tech Student  
Machine Learning & Remote Sensing Enthusiast  

GitHub: https://github.com/SaiAbhiYaswanth