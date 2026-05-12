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







#to setup on windows
pip install flask joblib numpy pandas scikit-learn
pip install matplotlib seaborn notebook
python app.py



# Terminal 1 (keep running)
python app.py

# Terminal 2 (run once)
python smoke_test.py
