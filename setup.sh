echo "--------------------------------"
echo "Set up environment"
echo "--------------------------------"

# Set up directories and environment variables
export TRANSFORMERS_CACHE=~/scratch/GenAI-Local-Deployment/model_source_files/.cache/huggingface
export HF_HOME=~/scratch/GenAI-Local-Deployment/model_source_files/.cache/huggingface
export PROJECT_DIRECTORY=~/scratch/GenAI-Local-Deployment

#Load environmental modules
module load python/3.10
module load StdEnv/2023  gcc/12.3  openmpi/4.1.5  cuda/12.2
module load arrow/15.0.1
module load faiss/1.7.4

# Set up the Python virtual environment
virtualenv --no-download $PROJECT_DIRECTORY/venv
source $PROJECT_DIRECTORY/venv/bin/activate

# Install required Python packages
pip install datasets --no-index
pip install transformers --no-index
pip install torch --no-index
pip install beir --no-index
pip install pandas --no-index
pip install numpy --no-index
pip install bitsandbytes --no-index
pip install flash_attn --no-index
pip install scipy --no-index
pip install sentencepiece --no-index
pip install accelerate>=0.26.0 --no-index
pip install --no-index pandas scikit_learn matplotlib seaborn
pip install --no-index tensorflow jupyterlab
pip install --no-index ipykernel

echo "--------------------------------"
echo "Environment setup complete."
echo "--------------------------------"