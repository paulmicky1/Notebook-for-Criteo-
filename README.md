# Criteo Campaign Analysis - CTR Prediction & Optimization

A comprehensive data analysis and machine learning project for analyzing Criteo display advertising campaigns, predicting click-through rates (CTR), and providing actionable business insights for campaign optimization.

## 📊 Project Overview

This project analyzes 100,000 display advertising impressions to understand user behavior patterns and predict click-through rates using machine learning. The analysis provides data-driven recommendations for optimizing campaign performance across devices, time periods, and bidding strategies.

### Key Objectives

- **Analyze** display advertising campaign performance across multiple dimensions
- **Predict** click-through rates using Random Forest classification
- **Identify** key factors influencing user engagement
- **Generate** actionable business recommendations for campaign optimization
- **Automate** professional presentation creation with insights and visualizations

## ✨ Features

### Data Analysis
- ✅ Comprehensive CTR analysis by device type, hour of day, and site category
- ✅ Feature engineering for enhanced predictive modeling
- ✅ Statistical analysis of campaign performance metrics
- ✅ Interactive visualizations with Matplotlib and Seaborn

### Machine Learning
- ✅ Random Forest classifier for CTR prediction
- ✅ ROC AUC evaluation and model performance metrics
- ✅ Feature importance analysis
- ✅ Stratified train-test split for balanced evaluation

### Business Intelligence
- ✅ Device optimization recommendations
- ✅ Peak hour timing strategies
- ✅ Dynamic bidding recommendations
- ✅ Projected revenue impact calculations

### Automated Reporting
- ✅ Professional PowerPoint presentation generation
- ✅ CSV exports for predictions and summary reports
- ✅ Customizable slides with charts and metrics

## 🚀 Getting Started

### Prerequisites

- Python 3.14 or higher
- pip or uv package manager

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Notebook-for-Criteo-
```

2. **Install dependencies**

Using `uv` (recommended):
```bash
uv sync
```

Or using `pip`:
```bash
pip install -r requirements.txt
```

### Dependencies

Key libraries used in this project:
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Presentation**: python-pptx
- **Big Data** (optional): pyspark, findspark
- **Web Scraping** (optional): scrapy, selenium, beautifulsoup4

Full dependency list available in `pyproject.toml`.

## 📖 Usage

### Running the Analysis Notebook

1. Launch Jupyter:
```bash
jupyter notebook
```

2. Open `criteo_analysis.ipynb` and run all cells to:
   - Load and explore the dataset
   - Perform feature engineering
   - Train machine learning models
   - Generate visualizations and insights
   - Export results

### Generating the Presentation

Run the presentation creation script:
```bash
python create_presentation.py
```

This will generate `Criteo_Campaign_Analysis_Presentation.pptx` with 13 professional slides including:
- Executive summary
- Dataset overview
- Performance metrics by device, hour, and category
- Machine learning model results
- Feature importance analysis
- Business insights and recommendations
- Implementation roadmap

### Running the Main Script

```bash
python main.py
```

## 📁 Project Structure

```
Notebook-for-Criteo-/
│
├── criteo_analysis.ipynb              # Main analysis notebook
├── create_presentation.py             # PowerPoint generation script
├── main.py                            # Main entry point
│
├── datasets/                          # Data directory
│   ├── advertising_and_sales_clean.csv
│   ├── Advertising_Data.csv
│   └── Advertising.csv
│
├── criteo_predictions.csv             # Model predictions output
├── criteo_summary_report.csv          # Summary metrics
├── Criteo_Campaign_Analysis_Presentation.pptx  # Generated presentation
│
├── pyproject.toml                     # Project configuration and dependencies
├── uv.lock                            # Dependency lock file
├── LICENSE                            # Project license
└── README.md                          # This file
```

## 📈 Key Results & Insights

### Performance Metrics

- **Overall CTR**: 7.86%
- **Total Impressions**: 100,000
- **Total Clicks**: 7,863
- **Model ROC AUC**: 0.5553

### Device Performance

| Device  | CTR   | Performance vs Desktop |
|---------|-------|------------------------|
| Mobile  | 8.97% | +41.9% better          |
| Desktop | 6.32% | baseline               |
| Tablet  | 5.89% | -6.8% worse            |

### Peak Hours Analysis

- **Peak Hours** (9-11 AM, 7-9 PM): 9.30% CTR
- **Off-Peak Hours**: 7.39% CTR
- **Performance Improvement**: +25.9% during peak hours

### Bidding Strategy

- **High Bid** (>$3.00): 9.26% CTR
- **Low Bid** (<$3.00): 6.94% CTR
- **ROI Optimization**: Strategic bidding increases CTR by +33.4%

### Feature Importance (Top 5)

1. **Bid Price** (20.4%)
2. **User Engagement** (12.6%)
3. **Creative Aspect Ratio** (10.0%)
4. **Hour of Day** (9.1%)
5. **Impression Count** (8.8%)

## 💡 Recommendations

### Immediate Actions (Week 1-2)
1. **Increase mobile ad spend by 15-20%** to capitalize on superior performance
2. **Concentrate 60% of budget during peak hours** (9-11 AM, 7-9 PM)
3. **Begin A/B testing** with optimized bid ranges ($2.50-$4.00)

### Model Deployment (Week 3-4)
4. **Deploy Random Forest model** to production for real-time optimization
5. **Set up monitoring dashboard** for continuous performance tracking
6. **Train team** on new optimization tools and processes

### Long-term Optimization (Month 2+)
7. **Fine-tune model parameters** based on live campaign results
8. **Expand to additional segments** and campaign types
9. **Implement continuous improvement** cycle with monthly reviews

### Projected Impact
- **CTR Improvement**: +25-30%
- **Revenue Increase**: +25-30%
- **Campaign Efficiency**: +20-25%

## 🛠️ Technologies Used

- **Python 3.14+**
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning models
- **Matplotlib & Seaborn** - Data visualization
- **python-pptx** - Presentation generation
- **Jupyter Notebook** - Interactive development environment

## 📊 Visualizations

The project includes various visualizations:
- CTR performance by device type (bar charts)
- Hourly CTR trends (line charts with peak hour highlights)
- Site category performance (horizontal bar charts)
- ROC curves for model evaluation
- Feature importance charts
- Confusion matrices

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the terms specified in the LICENSE file.

## 👤 Author

Paul M.

## 🙏 Acknowledgments

- Criteo for the dataset inspiration
- scikit-learn community for machine learning tools
- Jupyter Project for the interactive development environment

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This project uses synthetic data generated to match Criteo dataset characteristics. For production use, replace with actual campaign data from your advertising platform.

