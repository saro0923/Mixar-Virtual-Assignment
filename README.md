# 🧩 Mesh Normalization, Quantization, and Error Analysis 
*A comprehensive Python pipeline for mesh normalization, quantization, reconstruction, error analysis, and adaptive quantization with advanced 2D/3D visualizations.*

---

## 📋 Features
✅ **Mesh Normalization:** Min–Max and Unit Sphere normalization methods  
✅ **Quantization:** Configurable bin-based quantization (default: 1024 bins)  
✅ **Reconstruction:** Complete dequantization and denormalization pipeline  
✅ **Error Analysis:** Per-axis and global MAE/MSE metrics  
✅ **2D Visualizations:** Bar charts and histograms of quantization errors  
✅ **3D Visualizations:** Color-mapped error meshes with Open3D  
✅ **Bonus:** Adaptive Quantization — density-based adaptive bin allocation  
✅ **Bonus:** Transform Invariance — rotation and translation testing  

---

## 📁 Project Structure

```bash
mesh_preprocessing/
├── data/ # Input OBJ meshes
│   ├── Branch.obj
│   ├── Cylinder.obj
│   ├── Explosive.obj
│   ├── Fence.obj
│   ├── Girl.obj
│   ├── Person.obj
│   ├── Table.obj
│   └── Talwar.obj
├── outputs/ # Generated outputs
│   ├── stats/ # Mesh statistics JSON files
│   ├── quantized/ # Quantized mesh data (NPZ)
│   ├── recon/ # Reconstructed OBJ meshes + colored PLY
│   ├── plots/ # 2D error visualizations (PNG)
│   ├── adaptive_quantization/ # Adaptive quantization results
│   └── error_summary.csv # Aggregated error metrics
├── src/
│   ├── main.py # Pipeline orchestration
│   └── utils.py # Helper functions
├── requirements.txt
└── README.md

🚀 Installation
1️⃣ Clone or download the project:
git clone [https://github.com/saro0923/Mixar-Virtual-Assignment.git](https://github.com/saro0923/Mixar-Virtual-Assignment.git)
cd Mixar-Virtual-Assignment
2️⃣ Install dependencies:
pip install -r requirements.txt
3️⃣ Prepare your data: Place all .obj mesh files inside the data/ directory.

💻 Usage
▶️ Basic Pipeline (Min–Max + Unit Sphere)
ython src/main.py --input_dir data --output_dir outputs --methods minmax,unitsphere --n_bins 1024
🧠 With Adaptive Quantization (Bonus)
python src/main.py --input_dir data --output_dir outputs --methods minmax,unitsphere --n_bins 1024 --bonus adaptive
⚙️ Custom Configuration
python src/main.py \
  --input_dir data \
  --output_dir my_outputs \
  --methods unitsphere \
  --n_bins 2048 \
  --bonus adaptive
🔬 Pipeline Steps
1️⃣ Load & Inspect

Loads .obj files using trimesh

Extracts vertices (Nx3 NumPy array)

Computes per-axis min, max, mean, std

Saves stats to outputs/stats/<mesh>_stats.json
2️⃣ Normalization

Min–Max:

normalized = (v - v_min) / (v_max - v_min)

Scales each axis independently to [0, 1]

Unit Sphere:

centered = v - centroid

normalized = centered / max_distance

Centers mesh and scales to unit sphere.

3️⃣ Quantization

quantized = floor(normalized * (n_bins - 1))

Maps continuous [0, 1] values to discrete bins.

4️⃣ Reconstruction

Dequantize: v = quantized / (n_bins - 1)

Denormalize using saved metadata.

Saves reconstructed mesh as .obj.
5️⃣ Error Analysis

Computes per-axis & global MAE / MSE

Generates error plots per axis

6️⃣ Visualization

2D (Matplotlib):

Bar chart: MSE/MAE per axis

Histogram: Error magnitude distribution
3D (Open3D):

Color-mapped mesh (Blue = low error, Red = high error)
7️⃣ Adaptive Quantization (Bonus)

Applies random rotation + translation

Computes vertex density (k-NN based)

Allocates bins adaptively $\rightarrow$ more bins for dense regions

Compares uniform vs. adaptive quantization

Output Files
outputs/
├── stats/
│   └── Branch_stats.json
├── quantized/
│   ├── Branch_minmax_quantized.npz
│   └── Branch_unitsphere_quantized.npz
├── recon/
│   ├── Branch_minmax_recon.obj
│   ├── Branch_minmax_error_color.ply
│   ├── Branch_unitsphere_recon.obj
│   └── Branch_unitsphere_error_color.ply
├── plots/
│   ├── Branch_minmax_error_bars.png
│   ├── Branch_minmax_error_hist.png
│   ├── Branch_unitsphere_error_bars.png
│   └── Branch_unitsphere_error_hist.png
├── adaptive_quantization/
│   ├── Branch_density_vs_error.png
│   ├── Branch_comparison.png
│   └── adaptive_summary.csv
└── error_summary.csv
Here is the complete content for your README.md file, formatted with all the sections, code blocks, and tables you provided.You can copy and paste this directly into the README.md file in your GitHub repository.Markdown# 🧩 3D Mesh Preprocessing Pipeline
*A comprehensive Python pipeline for mesh normalization, quantization, reconstruction, error analysis, and adaptive quantization with advanced 2D/3D visualizations.*

---

## 📋 Features
✅ **Mesh Normalization:** Min–Max and Unit Sphere normalization methods  
✅ **Quantization:** Configurable bin-based quantization (default: 1024 bins)  
✅ **Reconstruction:** Complete dequantization and denormalization pipeline  
✅ **Error Analysis:** Per-axis and global MAE/MSE metrics  
✅ **2D Visualizations:** Bar charts and histograms of quantization errors  
✅ **3D Visualizations:** Color-mapped error meshes with Open3D  
✅ **Bonus:** Adaptive Quantization — density-based adaptive bin allocation  
✅ **Bonus:** Transform Invariance — rotation and translation testing  

---

## 📁 Project Structure

```bash
mesh_preprocessing/
├── data/ # Input OBJ meshes
│   ├── Branch.obj
│   ├── Cylinder.obj
│   ├── Explosive.obj
│   ├── Fence.obj
│   ├── Girl.obj
│   ├── Person.obj
│   ├── Table.obj
│   └── Talwar.obj
├── outputs/ # Generated outputs
│   ├── stats/ # Mesh statistics JSON files
│   ├── quantized/ # Quantized mesh data (NPZ)
│   ├── recon/ # Reconstructed OBJ meshes + colored PLY
│   ├── plots/ # 2D error visualizations (PNG)
│   ├── adaptive_quantization/ # Adaptive quantization results
│   └── error_summary.csv # Aggregated error metrics
├── src/
│   ├── main.py # Pipeline orchestration
│   └── utils.py # Helper functions
├── requirements.txt
└── README.md
🚀 Installation1️⃣ Clone or download the project:Bashgit clone [https://github.com/saro0923/Mixar-Virtual-Assignment.git](https://github.com/saro0923/Mixar-Virtual-Assignment.git)
cd Mixar-Virtual-Assignment
2️⃣ Install dependencies:Bashpip install -r requirements.txt
3️⃣ Prepare your data:Place all .obj mesh files inside the data/ directory.💻 Usage▶️ Basic Pipeline (Min–Max + Unit Sphere)Bashpython src/main.py --input_dir data --output_dir outputs --methods minmax,unitsphere --n_bins 1024
🧠 With Adaptive Quantization (Bonus)Bashpython src/main.py --input_dir data --output_dir outputs --methods minmax,unitsphere --n_bins 1024 --bonus adaptive
⚙️ Custom ConfigurationBashpython src/main.py \
  --input_dir data \
  --output_dir my_outputs \
  --methods unitsphere \
  --n_bins 2048 \
  --bonus adaptive
🔬 Pipeline Steps1️⃣ Load & InspectLoads .obj files using trimeshExtracts vertices (Nx3 NumPy array)Computes per-axis min, max, mean, stdSaves stats to outputs/stats/<mesh>_stats.json2️⃣ NormalizationMin–Max:normalized = (v - v_min) / (v_max - v_min)Scales each axis independently to [0, 1]Unit Sphere:centered = v - centroidnormalized = centered / max_distanceCenters mesh and scales to unit sphere.3️⃣ Quantizationquantized = floor(normalized * (n_bins - 1))Maps continuous [0, 1] values to discrete bins.4️⃣ ReconstructionDequantize: v = quantized / (n_bins - 1)Denormalize using saved metadata.Saves reconstructed mesh as .obj.5️⃣ Error AnalysisComputes per-axis & global MAE / MSEGenerates error plots per axis6️⃣ Visualization2D (Matplotlib):Bar chart: MSE/MAE per axisHistogram: Error magnitude distribution3D (Open3D):Color-mapped mesh (Blue = low error, Red = high error)7️⃣ Adaptive Quantization (Bonus)Applies random rotation + translationComputes vertex density (k-NN based)Allocates bins adaptively $\rightarrow$ more bins for dense regionsCompares uniform vs. adaptive quantization📊 Output FilesBashoutputs/
├── stats/
│   └── Branch_stats.json
├── quantized/
│   ├── Branch_minmax_quantized.npz
│   └── Branch_unitsphere_quantized.npz
├── recon/
│   ├── Branch_minmax_recon.obj
│   ├── Branch_minmax_error_color.ply
│   ├── Branch_unitsphere_recon.obj
│   └── Branch_unitsphere_error_color.ply
├── plots/
│   ├── Branch_minmax_error_bars.png
│   ├── Branch_minmax_error_hist.png
│   ├── Branch_unitsphere_error_bars.png
│   └── Branch_unitsphere_error_hist.png
├── adaptive_quantization/
│   ├── Branch_density_vs_error.png
│   ├── Branch_comparison.png
│   └── adaptive_summary.csv
└── error_summary.csv

📈 Results Interpretation

Error Summary Example
Mesh,Method,MSE_Total,MAE_Total
Branch,Min–Max,1.33e-04,1.00e-02
Branch,UnitSphere,1.05e-04,8.93e-03
✅ Lower MSE/MAE $\rightarrow$ better reconstruction
✅ Adaptive Quantization improves dense mesh region accuracy by 10–30%

🧪 Key Functions
Function,Purpose
load_mesh(),Load OBJ file using trimesh
compute_stats(),Calculate vertex statistics
normalize_minmax(),Min–Max normalization
normalize_unitsphere(),Unit sphere normalization
quantize(),Bin-based quantization
dequantize(),Reverse quantization
compute_error_metrics(),Calculate MAE/MSE
plot_error_charts(),Generate 2D plots
visualize_error_mesh(),Create 3D colored mesh
compute_vertex_density(),k-NN density computation
adaptive_quantize(),Density-based quantization

🎓 Technical Insights
Quantization Theory

Pros: Compression, speed

Cons: Small accuracy loss

Trade-off: More bins = higher precision

Adaptive Quantization

Allocates resources intelligently:

Dense regions → More bins

Sparse regions → Fewer bins
→ Improves accuracy/storage efficiency.

Error Metrics

MAE: Average absolute difference

MSE: Penalizes large errors more heavily

🧰 Troubleshooting

Error: ImportError: No module named 'trimesh'
✅ Run: pip install trimesh

Error: No 3D visualization window
✅ Open3D runs headless; view saved .ply in Blender/MeshLab.

Memory issue:
✅ Reduce bins to 512 or process meshes one by one.

📚 Dependencies

numpy

trimesh

open3d

matplotlib

scipy

pandas

🎯 Future Enhancements

Support for .STL, .PLY, .OFF

GPU acceleration for large meshes

Octree-based adaptive quantization

Interactive visualization (PyVista)

Multi-threaded batch processing

👤 Author

Saravanan S 
💻 Python Developer 
📅 November 2025
📂 Mixar Virtual Assignment – Recruitment Round 1

🙏 Acknowledgments

Built using:
NumPy, Trimesh, Open3D, Matplotlib, SciPy, and Pandas


---

