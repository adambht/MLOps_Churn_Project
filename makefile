# Define variables
PYTHON=python3
VENV=venv
ACTIVATE=$(VENV)/bin/activate
REQUIREMENTS=requirements.txt
DATASET=churn_combined.csv
MODEL=model.joblib
STREAMLIT_PORT=8501
STREAMLIT_URL=http://127.0.0.1:$(STREAMLIT_PORT)

.PHONY: all taktak install setup data train test deploy clean notebook evaluate streamlit_run open_streamlit test_streamlit

# Installation des dépendances (sans venv)
install:
	@echo ""
	@echo "🔶--> Installation des dépendances :"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r $(REQUIREMENTS)
	@echo "🆗 Installation terminee !!!"
	@echo ""

# taktak:
# Just for testing the makefile with prepare data, train, and
# evaluate and deploy without each time running the clean and the setup
taktak: help data train evaluate deploy

# 🚀 STEP 0 : Executer tout:
all: help clean setup test lint data train evaluate deploy streamlit_run
# In the same terminal: type make deploy
# In a terminal: type make test_streamlit

# STEP 1 : Help Command:
help:
	@echo ""
	@echo "🔵 [STEP 1] Commandes disponibles :"
	@echo " 1- make clean : Delete all temporary files."
	@echo " 2- make setup : Venv config and install dependencies."
	@echo " 3- make test : Verifies VENV."
	@echo " 4- make lint : Code quality (formatting, security)."
	@echo " 5- make data : Prepare data for training."
	@echo " 6- make train : Train the model."
	@echo " 7- make evaluate : Model testing."
	@echo " 8- make deploy   : Deploy the model."
	@echo " 9- make notebook : Start Jupyter Notebook."
	@echo "10- make streamlit_run : Run Streamlit."
	@echo "11- make open_streamlit : Automatically open Streamlit in browser."
	@echo ""

# STEP 2 : Nettoyage des fichiers:
clean:
	@echo ""
	@echo "🔵 [STEP 2] Cleaning up temporary files:"
	@rm -rf venv
	@find . -type f -name "*.installed" -exec rm -f {} +
	@echo "🆗 Cleaning completed."
	@echo ""

# STEP 3 : VENV + INSTALLATION:
setup:
	@echo ""
	@echo "🔵 [STEP 3] Setting up VENV + Dependencies:"
	@if [ ! -d "$(VENV)" ]; then $(PYTHON) -m venv $(VENV); fi
	@$(VENV)/bin/pip install --upgrade pip
	@$(VENV)/bin/pip install -r $(REQUIREMENTS)
	@echo "🆗 Setup completed."
	@echo ""

# STEP 5 : Check code quality, formatting, security:
lint:
	@echo ""
	@echo "🔵 [STEP 5] Checking code quality (formatting, security):"
	@. $(VENV)/bin/activate && black main.py model_pipeline.py
	@. $(VENV)/bin/activate && flake8 main.py model_pipeline.py
	@. $(VENV)/bin/activate && bandit -r main.py model_pipeline.py
	@echo "🆗 Code formatting, quality, and security checks completed."
	@echo ""

# STEP 6 : Prepare the data:
data:
	@echo ""
	@echo "🔵 [STEP 6] Preparing data:"
	@bash -c "source $(ACTIVATE) && $(PYTHON) main.py --prepare"
	@echo "🆗 Data prepared."
	@echo ""

# STEP 7 : Train the model:
train:
	@echo ""
	@echo "🔵 [STEP 7] Training model:"
	@bash -c "source $(ACTIVATE) && $(PYTHON) main.py --train"
	@echo "🆗 Model trained."
	@echo ""

# STEP 8 : Evaluate the model:
evaluate:
	@echo ""
	@echo "🔵 [STEP 8] Evaluating model:"
	@bash -c "source $(ACTIVATE) && $(PYTHON) main.py --evaluate"
	@echo "🆗 Model evaluated."
	@echo ""

# STEP 9 : Deploy the model:
deploy:
	@echo ""
	@echo "🔵 [STEP 9] Deploying model:"
	@bash -c "source $(ACTIVATE) && $(PYTHON) main.py --deploy"
	@bash -c "source $(ACTIVATE) && $(PYTHON) main.py --predict"
	@echo "🆗 Model deployed."
	@echo ""

# STEP 10 : Start Streamlit app:
streamlit_run:
	@echo ""
	@echo "🔵 [STEP 10] Running Streamlit app:"
	@streamlit run app_streamlit.py --server.port $(STREAMLIT_PORT)
	@echo "🆗 Streamlit app is running."
	@echo ""











# STEP 4 : Simple test:
#test:
#	@echo ""
#	@echo "🔵 [STEP 4] Testing VENV:"
#	@$(PYTHON) test_environment.py
#	@echo ""