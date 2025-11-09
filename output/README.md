# Mixar Virtual Assignment - Mesh Normalization, Quantization & Error Analysis

## 📘 Overview
This project processes 3D mesh (.obj) files by performing:
- **Normalization:** Min-Max and Unit-Sphere scaling
- **Quantization:** 1024 uniform bins per axis
- **Reconstruction:** Dequantization and Denormalization
- **Error Analysis:** MSE and MAE per-axis, plotted as charts

The outputs include normalized/reconstructed meshes and error plots.

---

## 🧠 Steps to Run
1. Place `.obj` files into `input_meshes/`
2. Activate virtual environment:
   ```bash
   venv\Scripts\activate
Run the script:

python mesh_preprocess.py --input_dir input_meshes --output_dir output_all --bins 1024


Outputs are saved in output_all/
View summary_errors.csv for results.

🧩 Dependencies
numpy
trimesh
matplotlib


Install via:

pip install numpy trimesh matplotlib

🧾 Results Summary
Mesh	Vertices	MinMax MSE	MinMax MAE	UnitSphere MSE	UnitSphere MAE
cube	24	0.000000	0.000000	0.000001	0.000846
teapot	529	0.005668	0.062596	0.010617	0.087349
📊 Visual Outputs

Check output_all/<meshname>/plots/ for:

mse_axis.png

hist_mm.png

hist_us.png

👨‍💻 Author

Saravanan S Sekar
Completed as part of Mixar Virtual Assignment – Round 1 (Recruitment Process)