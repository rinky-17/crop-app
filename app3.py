from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd
import shap
import plotly.express as px

app = Flask(__name__)

# =========================
# LOAD FILES
# =========================
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
metrics = pickle.load(open("metrics.pkl", "rb"))

explainer = shap.TreeExplainer(model)

# =========================
# FEATURE RANGES
# =========================
feature_ranges = {
    "Nitrogen": (0,140),
    "Phosphorus": (5,145),
    "Potassium": (5,205),
    "Temperature": (8,43),
    "Humidity": (14,100),
    "Ph": (3.5,9.5),
    "Rainfall": (20,300)
}

# =========================
# ENCODING DICTS
# =========================
soil_dict = {"Clay":0,"Sandy":1,"Loamy":2,"Silty":3,"Peaty":4}
season_dict = {"Summer":0,"Winter":1,"Monsoon":2}

location_dict = {
    "Andhra Pradesh - Visakhapatnam":0,
    "Telangana - Hyderabad":1,
    "Tamil Nadu - Coimbatore":2,
    "Karnataka - Mysuru":3,
    "Maharashtra - Pune":4,
    "Punjab - Ludhiana":5,
    "Uttar Pradesh - Lucknow":6,
    "West Bengal - Kolkata":7
}

# =========================
# CROPS
# =========================
crop_dict = {
    0:"Rice",1:"Maize",2:"Jute",3:"Cotton",4:"Coconut",5:"Papaya",
    6:"Orange",7:"Apple",8:"Muskmelon",9:"Watermelon",10:"Grapes",
    11:"Mango",12:"Banana",13:"Pomegranate",14:"Lentil",
    15:"Blackgram",16:"Mungbean",17:"Mothbeans",18:"Pigeonpeas",
    19:"Kidneybeans",20:"Chickpea",21:"Coffee"
}

# =========================
# CROP TRANSLATIONS
# =========================
crop_translations = {
    "hi": {"Rice":"चावल","Maize":"मक्का","Jute":"पटसन","Cotton":"कपास","Coconut":"नारियल","Papaya":"पपीता","Orange":"संतरा","Apple":"सेब","Muskmelon":"खरबूजा","Watermelon":"तरबूज","Grapes":"अंगूर","Mango":"आम","Banana":"केला","Pomegranate":"अनार","Lentil":"मसूर","Blackgram":"उड़द","Mungbean":"मूंग","Mothbeans":"मटकी","Pigeonpeas":"अरहर","Kidneybeans":"राजमा","Chickpea":"चना","Coffee":"कॉफी"},
    "te": {"Rice":"బియ్యం","Maize":"మొక్కజొన్న","Jute":"జూట్","Cotton":"పత్తి","Coconut":"కొబ్బరి","Papaya":"బొప్పాయి","Orange":"నారింజ","Apple":"ఆపిల్","Muskmelon":"కర్బూజ","Watermelon":"పుచ్చకాయ","Grapes":"ద్రాక్ష","Mango":"మామిడి","Banana":"అరటి","Pomegranate":"దానిమ్మ","Lentil":"పప్పు","Blackgram":"ఉలవలు","Mungbean":"పెసలు","Mothbeans":"మోత్ బీన్స్","Pigeonpeas":"కందులు","Kidneybeans":"రాజ్మా","Chickpea":"సెనగలు","Coffee":"కాఫీ"},
    "ta": {"Rice":"அரிசி","Maize":"மக்காச்சோளம்","Jute":"ஜூட்","Cotton":"பருத்தி","Coconut":"தேங்காய்","Papaya":"பப்பாளி","Orange":"ஆரஞ்சு","Apple":"ஆப்பிள்","Muskmelon":"முலாம்பழம்","Watermelon":"தர்பூசணி","Grapes":"திராட்சை","Mango":"மாம்பழம்","Banana":"வாழை","Pomegranate":"மாதுளை","Lentil":"பருப்பு","Blackgram":"உளுந்து","Mungbean":"பாசிப்பயறு","Mothbeans":"மொத் பீன்ஸ்","Pigeonpeas":"துவரம்","Kidneybeans":"ராஜ்மா","Chickpea":"கொண்டைக்கடலை","Coffee":"காபி"},
    "kn": {"Rice":"ಅಕ್ಕಿ","Maize":"ಮೆಕ್ಕೆಜೋಳ","Jute":"ಜ್ಯೂಟ್","Cotton":"ಹತ್ತಿ","Coconut":"ತೆಂಗಿನಕಾಯಿ","Papaya":"ಪಪ್ಪಾಯಿ","Orange":"ಕಿತ್ತಳೆ","Apple":"ಸೇಬು","Muskmelon":"ಕರ್ಬೂಜ","Watermelon":"ಕಲ್ಲಂಗಡಿ","Grapes":"ದ್ರಾಕ್ಷಿ","Mango":"ಮಾವು","Banana":"ಬಾಳೆಹಣ್ಣು","Pomegranate":"ದಾಳಿಂಬೆ","Lentil":"ಮಸೂರು","Blackgram":"ಉದ್ದಿನಕಾಳು","Mungbean":"ಹೆಸರುಕಾಳು","Mothbeans":"ಮೋತ್ ಬೀನ್ಸ್","Pigeonpeas":"ತೊಗರಿ","Kidneybeans":"ರಾಜ್ಮಾ","Chickpea":"ಕಡಲೆ","Coffee":"ಕಾಫಿ"}
}

# =========================
# FEATURE TRANSLATIONS
# =========================
feature_name_translations = {
    "hi":{"Nitrogen":"नाइट्रोजन","Phosphorus":"फॉस्फोरस","Potassium":"पोटैशियम","Temperature":"तापमान","Humidity":"आर्द्रता","pH":"pH","Rainfall":"वर्षा"},
    "te":{"Nitrogen":"నైట్రోజన్","Phosphorus":"ఫాస్ఫరస్","Potassium":"పోటాషియం","Temperature":"ఉష్ణోగ్రత","Humidity":"ఆర్ద్రత","pH":"pH","Rainfall":"వర్షపాతం"},
    "ta":{"Nitrogen":"நைட்ரஜன்","Phosphorus":"பாஸ்பரஸ்","Potassium":"பொட்டாசியம்","Temperature":"வெப்பநிலை","Humidity":"ஈரப்பதம்","pH":"pH","Rainfall":"மழை"},
    "kn":{"Nitrogen":"ನೈಟ್ರೋಜನ್","Phosphorus":"ಫಾಸ್ಫರಸ್","Potassium":"ಪೊಟ್ಯಾಸಿಯಂ","Temperature":"ತಾಪಮಾನ","Humidity":"ಆರ್ದ್ರತೆ","pH":"pH","Rainfall":"ಮಳೆಯ ಪ್ರಮಾಣ"}
}

# =========================
# TEXT TRANSLATIONS
# =========================
translations = {
    "en":{"result":"is the best crop to cultivate.","error":"Enter correct value","why":"Why this crop?","suggestion":"Suggestions","favorable":"Favorable","less":"Less favorable","optimal":"All conditions are optimal ✅","increase_k":"Increase Potassium","adjust_ph":"Adjust pH","improve_humidity":"Improve humidity"},
    "hi":{"result":"उगाने के लिए सबसे अच्छी फसल है।","error":"सही मान दर्ज करें","why":"यह फसल क्यों?","suggestion":"सुझाव","favorable":"अनुकूल","less":"कम अनुकूल","optimal":"सभी स्थितियां उत्तम हैं ✅","increase_k":"पोटैशियम बढ़ाएं","adjust_ph":"pH समायोजित करें","improve_humidity":"आर्द्रता सुधारें"},
    "te":{"result":"పండించడానికి ఉత్తమ పంట.","error":"సరైన విలువను నమోదు చేయండి","why":"ఈ పంట ఎందుకు?","suggestion":"సూచనలు","favorable":"అనుకూలం","less":"తక్కువ అనుకూలం","optimal":"అన్ని పరిస్థితులు సరైనవి ✅","increase_k":"పోటాషియం పెంచండి","adjust_ph":"pH సరిచేయండి","improve_humidity":"తేమను మెరుగుపరచండి"},
    "ta":{"result":"வளர்க்க சிறந்த பயிர்.","error":"சரியான மதிப்பை உள்ளிடவும்","why":"இந்த பயிர் ஏன்?","suggestion":"பரிந்துரைகள்","favorable":"சாதகமானது","less":"குறைவான சாதகமானது","optimal":"அனைத்து நிலைகளும் சிறந்தவை ✅","increase_k":"பொட்டாசியம் அதிகரிக்கவும்","adjust_ph":"pH ஐ சரிசெய்க","improve_humidity":"ஈரப்பதத்தை மேம்படுத்தவும்"},
    "kn":{"result":"ಬೆಳೆಯಲು ಅತ್ಯುತ್ತಮ ಬೆಳೆ.","error":"ಸರಿಯಾದ ಮೌಲ್ಯವನ್ನು ನಮೂದಿಸಿ","why":"ಈ ಬೆಳೆ ಏಕೆ?","suggestion":"ಸಲಹೆಗಳು","favorable":"ಅನುಕೂಲಕರ","less":"ಕಡಿಮೆ ಅನುಕೂಲಕರ","optimal":"ಎಲ್ಲಾ ಪರಿಸ್ಥಿತಿಗಳು ಉತ್ತಮವಾಗಿವೆ ✅","increase_k":"ಪೊಟ್ಯಾಸಿಯಂ ಹೆಚ್ಚಿಸಿ","adjust_ph":"pH ಸರಿಪಡಿಸಿ","improve_humidity":"ಆರ್ದ್ರತೆಯನ್ನು ಸುಧಾರಿಸಿ"}
}

# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return render_template("index3.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        lang = request.form.get('language','en')
        t = translations.get(lang, translations['en'])

        # INPUT
        values = {
            "Nitrogen": float(request.form['Nitrogen']),
            "Phosphorus": float(request.form['Phosphorus']),
            "Potassium": float(request.form['Potassium']),
            "Temperature": float(request.form['Temperature']),
            "Humidity": float(request.form['Humidity']),
            "Ph": float(request.form['Ph']),
            "Rainfall": float(request.form['Rainfall'])
        }

        # VALIDATION
        for k,v in values.items():
            mn,mx = feature_ranges[k]
            if not (mn <= v <= mx):
                return render_template("index3.html",
                    result=f"❌ {t['error']}",
                    top3_list=[], reason="", suggestion_text="",
                    graph=None, metrics=metrics, t=t)

        soil = soil_dict[request.form['Soil']]
        season = season_dict[request.form['Season']]
        location = location_dict[request.form['Location']]

        features = list(values.values()) + [soil,season,location]
        df = pd.DataFrame([features])

        final = sc.transform(ms.transform(df))

        # TOP 3
        probs = model.predict_proba(final)[0]
        top3_idx = np.argsort(probs)[-3:][::-1]

        top3_list=[]
        for i in top3_idx:
            crop = crop_dict[i]
            crop_name = crop_translations.get(lang, {}).get(crop, crop)
            conf = round(probs[i]*100,2)
            top3_list.append((crop_name, conf))

        top_crop = crop_dict[top3_idx[0]]
        top_crop = crop_translations.get(lang, {}).get(top_crop, top_crop)
        result = f"{top_crop} {t['result']}"

        # SHAP
        shap_values = explainer.shap_values(final)
        explanation = np.array(shap_values[0]).flatten()

        feature_names = list(values.keys())
        translated = [feature_name_translations.get(lang, {}).get(f, f) for f in feature_names]

        pairs = sorted(zip(translated, explanation), key=lambda x: abs(x[1]), reverse=True)[:3]

        pos=[f for f,v in pairs if v>0]
        neg=[f for f,v in pairs if v<0]

        reason=""
        if pos:
            reason += f"{t['favorable']}: " + ", ".join(pos)
        if neg:
            if reason: reason += " | "
            reason += f"{t['less']}: " + ", ".join(neg)

        suggestions=[]
        for f,v in zip(feature_names, explanation):
            if v<0:
                if f=="Potassium": suggestions.append(t["increase_k"])
                elif f=="Ph": suggestions.append(t["adjust_ph"])
                elif f=="Humidity": suggestions.append(t["improve_humidity"])

        suggestion_text = " | ".join(suggestions) if suggestions else t["optimal"]

        fig = px.bar(x=translated, y=model.feature_importances_[:7])
        graph = fig.to_html(full_html=False)

        

        return render_template("index3.html",
            result=result,
            top3_list=top3_list,
            reason=reason,
            suggestion_text=suggestion_text,
            graph=graph,
            metrics=metrics,
            t=t
        )

    except Exception as e:
        return render_template("index3.html",
            result=f"Error: {str(e)}",
            top3_list=[], reason="", suggestion_text="",
            graph=None, metrics=metrics,
            t=translations['en']
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 10000)
