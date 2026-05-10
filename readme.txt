https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?select=creditcard.csv


python -m notebook

jupyter

--- Setup on Kali Linux ---

1. Install system dependencies:
   sudo apt update
   sudo apt install -y python3-pip python3-venv

2. Create and activate a virtual environment:
   python3 -m venv venv
   source venv/bin/activate

3. Install all required Python packages:
   pip install pandas numpy matplotlib seaborn scikit-learn notebook ipykernel

4. Register the Jupyter kernel:
   python -m ipykernel install --user --name=fraud_detection --display-name "Python (fraud_detection)"

5. Launch Jupyter Notebook:
   python -m notebook
   (or simply: jupyter notebook)

Once the notebook opens:
   - Go to Kernel -> Change kernel
   - Select "Python (fraud_detection)"

When you are done, deactivate the virtual environment:
   deactivate

1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28




Normal transaction
Time: 0
V1–V28: -1.359807,-0.072781,2.536347,1.378155,-0.338321,0.462388,0.239599,0.098698,0.363787,0.090794,-0.551600,-0.617801,-0.991390,-0.311169,1.468177,-0.470401,0.207971,0.025791,0.403993,0.251412,-0.018307,0.277838,-0.110474,0.066928,0.128539,-0.189115,0.133558,-0.021053
Amount: 149.62
Expected: Normal (probability low)

Fraud transaction
Take one from your dataset where Class=1, or create a synthetic one with extreme V values.

This makes it easy to demo the app during your evaluation.




#to setup on windows
pip install flask joblib numpy pandas scikit-learn
pip install matplotlib seaborn notebook
python app.py



# Terminal 1 (keep running)
python app.py

# Terminal 2 (run once)
python smoke_test.py
