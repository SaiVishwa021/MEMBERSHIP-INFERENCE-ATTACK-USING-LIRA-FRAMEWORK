# How to Reproduce Results
## 1. Clone the Repository
```
git clone [<your-repo-link>](https://github.com/SaiVishwa021/MEMBERSHIP-INFERENCE-ATTACK-USING-LIRA-FRAMEWORK/tree/main)
cd mia_attack.py
```
## 2. Install Dependencies
Make sure you have Python ≥ 3.9 and install required libraries:
```
pip install torch torchvision numpy pandas scipy requests
```
## 3. Download Dataset and Model
Run:
```
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt"
```
## 4. Update File Paths
In mia_attack.py, update:
```
PUB_PATH   = "path/to/pub.pt"
PRIV_PATH  = "path/to/priv.pt"
MODEL_PATH = "path/to/model.pt"
OUTPUT_CSV = Path("path/to/submission.csv")
SHADOW_DIR = Path("path/to/shadow_models")
```
Also replace: 
```
API_KEY = "YOUR_API_KEY"
```
## 5. Run the Attack
```
python mia_attack.py
```
