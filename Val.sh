set -e  # Stop on the first error.

PROJECT_ROOT="/Users/gyh/Desktop/Harahiroshi/Competitions/iGEM_2026/DryLabWork/ML4OCRP"
DATA_DIR="$PROJECT_ROOT/Data"
MODEL_DIR="$PROJECT_ROOT/Model"
INSILICO_DIR="$PROJECT_ROOT/InSilicoValidation"

# Activate the conda environment.
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ocrp

# Generate the ratio and score tables.
cd "$INSILICO_DIR"

echo "Step 1: Generating ratio data..."
python fitting_ratio.py \

echo "Step 2: Generating score data..."
python fitting_score.py \

# Preprocess the tables into a per-id dictionary.
cd "$MODEL_DIR"

echo "Step 3: Preprocessing data..."
python preprocess_data.py

# Train the model.
echo "Step 4: Training model..."
python ocrp_model_train.py \

# Predict the optimum and the UCB recommendation.
echo "Step 5: Predicting and recommending..."
python predict_ratio_and_recommend_by_ucb.py \

# Query a time for each cell group.
echo "Step 6: Query per id..."
python query_per_id.py --fit_ode

# Compare each predicted time with the ODE.
cd "$INSILICO_DIR"
echo "Step 7: Chech actual ratio at (id,time)"
python get_actual_ratio.py

echo "Pipeline completed successfully!"