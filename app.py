{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "4b4c0fb9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ADMIN\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:347: InconsistentVersionWarning: Trying to unpickle estimator SVC from version 1.5.2 when using version 1.3.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\n",
      "https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations\n",
      "  warnings.warn(\n",
      "C:\\Users\\ADMIN\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:347: InconsistentVersionWarning: Trying to unpickle estimator LabelEncoder from version 1.5.2 when using version 1.3.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\n",
      "https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations\n",
      "  warnings.warn(\n",
      "C:\\Users\\ADMIN\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:347: InconsistentVersionWarning: Trying to unpickle estimator VotingClassifier from version 1.5.2 when using version 1.3.0. This might lead to breaking code or invalid results. Use at your own risk. For more info please refer to:\n",
      "https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:No training configuration found in the save file, so the model was *not* compiled. Compile it manually.\n",
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      "127.0.0.1 - - [26/Sep/2024 13:02:27] \"GET / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 3s 3s/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [26/Sep/2024 13:02:38] \"POST / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 177ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [26/Sep/2024 13:02:48] \"POST / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 160ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [26/Sep/2024 13:03:17] \"POST / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 184ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [26/Sep/2024 13:03:25] \"POST / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 151ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [26/Sep/2024 13:03:35] \"POST / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 168ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [26/Sep/2024 13:03:49] \"POST / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 167ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [26/Sep/2024 13:04:01] \"POST / HTTP/1.1\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1/1 [==============================] - 0s 181ms/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "127.0.0.1 - - [26/Sep/2024 13:04:17] \"POST / HTTP/1.1\" 200 -\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, request, render_template\n",
    "import joblib\n",
    "import numpy as np\n",
    "import cv2\n",
    "import os\n",
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "# Load the saved ensemble model\n",
    "ensemble_model = joblib.load('svm_xgb_ensemble.pkl')\n",
    "densenet_model_path = 'densenet201_extractor.h5'  # Update with your actual path\n",
    "densenet_extractor = load_model(densenet_model_path)\n",
    "\n",
    "# Create uploads folder if it doesn't exist\n",
    "if not os.path.exists('uploads'):\n",
    "    os.makedirs('uploads')\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/', methods=['GET', 'POST'])\n",
    "def home():\n",
    "    prediction = \"No prediction yet\"  # Default value for prediction\n",
    "    if request.method == 'POST':\n",
    "        # Get the uploaded file\n",
    "        file = request.files['file']\n",
    "        if file:\n",
    "            # Save the uploaded file temporarily\n",
    "            image_path = os.path.join('uploads', file.filename)\n",
    "            file.save(image_path)\n",
    "\n",
    "            # Preprocess the image\n",
    "            image = cv2.imread(image_path)\n",
    "            image = cv2.resize(image, (224, 224))  # Resize for DenseNet input\n",
    "            image_array = np.array(image) / 255.0  # Normalize the image\n",
    "            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension\n",
    "            features = densenet_extractor.predict(image_array)\n",
    "\n",
    "            # Make prediction with the ensemble model\n",
    "            prediction_index = ensemble_model.predict(features)[0]  # Get the prediction index\n",
    "\n",
    "            # Map the prediction to class labels\n",
    "            if prediction_index == 0:\n",
    "                prediction = \"Normal\"\n",
    "            elif prediction_index == 1:\n",
    "                prediction = \"Benign\"\n",
    "            elif prediction_index == 2:\n",
    "                prediction = \"Malignant\"\n",
    "\n",
    "            # Clean up: remove the uploaded file after processing\n",
    "            os.remove(image_path)\n",
    "\n",
    "    return render_template('index.html', prediction=prediction)\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(port=5000)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "918d4866",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
