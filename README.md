# Digital Histology Challenge
# Digital Histology Challenge

## 1. Introduction
**Helicobacter pylori** is a bacterium with high global prevalence, strongly linked to various gastrointestinal conditions, including gastric cancer. This makes early diagnosis and prevention critical. Traditional diagnosis through visual examination of histopathological samples stained with immunohistochemistry is labor-intensive and limited by image size and staining variability.

The aim of this project is to develop automated machine learning tools to detect the bacterium in these images. The project addresses challenges such as data size, imbalance, and complexity using advanced techniques like **attention mechanisms** and **generative models**. By focusing on patch-based classification and reproducibility evaluation across independent datasets, this project seeks to enhance precision, efficiency, and consistency in digital diagnosis of H. pylori.

### 1.1 Objectives

**Main Objective:**  
To develop automated methods for detecting the presence of H. pylori in histopathological images, improving accuracy, efficiency, and reproducibility in medical diagnosis.

**Specific Objectives:**  
- **Data Preparation**: Divide images into 256×256 pixel patches and apply augmentation to balance positive and negative samples.
- **Patch Classification**: Implement generative models to detect staining anomalies.
- **Attention Mechanisms**: Use attention-based techniques to identify relevant regions within patches.
- **Patient-Level Diagnosis**: Aggregate patch results to produce an overall diagnosis.
- **Validation and Testing**: Validate the models through cross-validation and test them on unseen data (HoldOut set).
- **Comparison of Methods**: Compare generative models and attention mechanisms to identify the most effective approach.

These tasks aim to tackle project challenges and ensure the development of a reliable diagnostic system for clinical use.

## 2. Methodology

The project workflow is divided into two complementary systems for detecting H. pylori. Each system applies specific methodologies for **patch classification** and **patient-level diagnosis**.

### **2.1 System 1: AutoEncoder-Based Approach**
This system uses an AutoEncoder to reconstruct negative patient patches (without H. pylori).  
**Steps**:
1. **AutoEncoder Training**: Train the model using the Cropped dataset, focusing on negative patches.
2. **Patch Classification**: Positive patches generate higher reconstruction errors. These errors are compared against an optimal threshold (calculated using the Annotated dataset) to classify patches as positive or negative.
3. **Patient Diagnosis**: Calculate the percentage of positive patches per patient. A threshold derived from the Cropped dataset determines if the patient is classified as positive or negative.

### **2.2 System 2: Attention Mechanism for Patient Diagnosis**
System 2 leverages the encoder from System 1 to extract patch features, applying an attention mechanism for direct patient-level diagnosis.  
**Steps**:
1. **Feature Extraction**: The encoder transforms patches into feature representations.
2. **Attention Training**: Train an attention mechanism model using the Cropped dataset to highlight the most relevant regions for global diagnosis.
3. **Patient Diagnosis**: The attention mechanism aggregates patch-level features to predict whether the patient is positive or negative.

### **2.3 Validation and Testing**
1. **Cross-Validation**: Validate both systems using the Annotated dataset to evaluate accuracy in patch classification and diagnosis.
2. **HoldOut Testing**: Test models on the independent HoldOut dataset to assess generalization and robustness.

The combination of these methodologies ensures a robust and efficient diagnostic system, addressing challenges related to imbalanced data and complex histological samples.


# Experimental Design

## 3.1 Dataset Description
Two datasets provided by the **Quirón Salut Dataset** were used for training, validation, and testing:

### 1. **Cropped Dataset**
This dataset consists of 256×256 pixel fragments (patches) extracted from gastric biopsy Whole Slide Images (WSIs) in TIFF format, focusing on diagnostically relevant regions.  
- **Purpose**:
  - Training the AutoEncoder for System 1 to reconstruct negative patches.
  - Training the attention mechanism in System 2.
- **Organization**: Grouped into folders by patient ID (PatID) and tissue section (PatID_Section#).

### 2. **Annotated Dataset**
Manually annotated by experts, this dataset contains 1,211 fragments, of which 161 are positive (containing H. pylori). Negative images were added to balance the dataset.  
- **Purpose**: Validation of fragment classification and diagnosis models.
- **Annotations**:
  - `-1`: Absence of H. pylori.
  - `1`: Presence of H. pylori.
  - `0`: Unclear image.

### 3. **HoldOut Dataset**
This independent dataset includes fragments from 116 previously unseen images, ensuring the system generalizes well.  
- **Purpose**: Testing system robustness and reproducibility.

## 3.2 Experiments and Metrics

### 3.2.1 **System 1: AutoEncoder-Based Patch Classification**
- **Hyperparameter Tuning**:  
  A random search optimized parameters like learning rate (0.0001–0.01), dropout rate (0.1–0.5), batch size (32–128), and epochs (5–20).
- **Patch Classification**:  
  The AutoEncoder was trained on negative fragments. Reconstruction error, calculated using the HSV color space, flagged positive patches by identifying red tonalities indicative of H. pylori.  
  Steps:
  1. Convert original and reconstructed images to HSV space.
  2. Extract the Hue channel.
  3. Detect red regions and compare pixel counts between original and reconstructed images.
  4. High error indicates likely presence of H. pylori.
  - **ROC Curve and AUC**: Used to determine the optimal error threshold for classifying fragments.
- **Patient Diagnosis**:  
  The proportion of positive patches was used to diagnose patients, with thresholds set via ROC analysis.
  
### 3.2.2 **System 2: Attention Mechanism for Patient Diagnosis**
- **Feature Extraction**:  
  The encoder from System 1 transformed patches into 64-dimensional feature vectors.
- **Attention Model**:  
  Features were aggregated using a tensor structure with:
  - **Feature Dim**: 64 (input size per fragment).
  - **Decomposition Space**: 128 (projection size for queries and keys).
- **Classifier**:  
  A fully connected network transformed the context vector into binary patient-level predictions.
  - Architecture: Linear layers (64 → 128 → 2 dimensions), ReLU activation, and 50% dropout.

### 3.2.3 Metrics
Both systems were evaluated with the following metrics:

1. **Precision**: Proportion of true positive predictions out of all positive predictions.  
   \( \text{Precision} = \frac{\text{TP}}{\text{TP + FP}} \)

2. **Recall (Sensitivity)**: Proportion of true positives correctly identified.  
   \( \text{Recall} = \frac{\text{TP}}{\text{TP + FN}} \)

3. **F1 Score**: Balance between Precision and Recall.  
   \( \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \)

4. **ROC Curve and AUC**:  
   - **True Positive Rate (TPR)**:  
     \( \text{TPR} = \frac{\text{TP}}{\text{TP + FN}} \)
   - **False Positive Rate (FPR)**:  
     \( \text{FPR} = \frac{\text{FP}}{\text{FP + TN}} \)

AUC values quantify model performance:
- \( \text{AUC} = 1 \): Perfect performance.
- \( \text{AUC} = 0.5 \): Random chance.
- \( \text{AUC} > 0.9 \): High-performing model.

## Summary
Cross-validation was performed using the Annotated dataset to compute average metrics and standard deviation, ensuring robust evaluation. Testing with the HoldOut dataset validated model generalization. Special focus was placed on maximizing recall, crucial in medical diagnostics to minimize false negatives and ensure accurate identification of H. pylori presence.

# Results

## 4.1 AutoEncoder Parameter Tuning
Using random search, the optimal AutoEncoder configuration identified is as follows:
- **Learning Rate**: 0.003939
- **Dropout Rate**: 0.265
- **Batch Size**: 32
- **Epochs**: 15
- **Final Loss**: 0.211186

The loss decreased consistently across epochs, indicating stable convergence and effective learning of image reconstruction for negative fragments.


## 4.2 Patch Classification
During cross-validation (20 folds), the AutoEncoder achieved an **AUC of 81%** for patch classification, indicating a good ability to distinguish between positive and negative patches.

### Threshold Selection:
- The optimal threshold for determining a positive patch was 210, consistently across folds.


## 4.3 Patient Diagnosis
Using the threshold from the Cropped Dataset, the optimal threshold for patient diagnosis was determined to be **0.48**, meaning a minimum of 48% positive patches is required to classify a patient as positive.  
- The best-performing fold achieved an **AUC of 92%**, reflecting excellent results.

## 4.4 System 1 Results
The cross-validation results for System 1 are summarized in the table below:

| **Metric**    | **Class 0**            | **Class 1**            |
|---------------|------------------------|------------------------|
| **Precision** | 0.65 ± 0.30            | 0.85 ± 0.31            |
| **Recall**    | 0.78 ± 0.35            | 0.75 ± 0.28            |
| **F1 Score**  | 0.85 ± 0.29            | 0.79 ± 0.30            |

The results demonstrate good performance, but the high standard deviation suggests potential variability due to outlier folds. Upon analysis:
- **Fold 2 and Fold 4** classified all cases as negative (Class 0), leading to skewed metrics.
- **Fold 5** classified all cases correctly except for one negative patient.

### HoldOut Dataset:
Results from the independent HoldOut dataset are as follows:

| **Metric**    | **Class 0** | **Class 1** |
|---------------|-------------|-------------|
| **Recall**    | 1.00        | 0.10        |
| **Precision** | 0.53        | 0.69        |
| **F1 Score**  | 1.00        | 0.19        |

Despite good metrics for Class 0, the results for Class 1 were suboptimal, indicating the model classified most patients as negative. This was confirmed by the confusion matrix. Further tests to adjust thresholds and exclude low-density patients slightly improved Class 1 performance but remained insufficient.

## 4.5 System 2 Results
Cross-validation results for System 2 (Attention Mechanism) are summarized below:

| **Metric**    | **Class 0**            | **Class 1**            |
|---------------|------------------------|------------------------|
| **Precision** | 0.47 ± 0.33            | 0.64 ± 0.42            |
| **Recall**    | 0.44 ± 0.36            | 0.58 ± 0.35            |
| **F1 Score**  | 0.40 ± 0.29            | 0.56 ± 0.36            |

The results suggest that the model performs better for positive patients (Class 1). However, the large standard deviation indicates variability or potential fold-specific issues. Further analysis to identify outliers was not completed due to time constraints.

## Key Observations:
1. **System 1** showed strong performance in cross-validation but struggled to generalize on the HoldOut dataset.
2. **System 2** showed potential in cross-validation but requires further tuning to reduce variability and improve generalization.
3. Outliers and data imbalance (e.g., low-density patients) significantly impacted the metrics.

Further work is required to enhance the robustness and reliability of both systems, especially for unseen datasets like HoldOut.

# Conclusions

In this work, we explored two main techniques to detect *Helicobacter pylori* in tissue images. While the results are promising, several areas for improvement were identified.

### Key Findings:
1. **System 1 (AutoEncoder)**:
   - Demonstrated effectiveness in identifying patterns in images and accurately reconstructing negative fragments.
   - Achieved an **AUC of 81%** in cross-validation for patch classification, showing a good ability to distinguish positive and negative fragments.
   - Achieved a **92% AUC** in patient diagnosis during cross-validation, indicating strong performance in detecting positive cases at the patient level.
   - However, some folds exhibited significant deviations, raising concerns about generalization in specific conditions.
   - Results on the **HoldOut dataset** were below expectations, likely due to non-optimal threshold configurations or limitations in the AutoEncoder’s setup.
   - A more in-depth comparison of the three possible AutoEncoder configurations could have provided insights into how each setup impacts performance.

2. **System 2 (Attention Mechanism)**:
   - Performance was suboptimal compared to System 1, with cross-validation metrics ranging between **0.4 and 0.63**, highlighting its limited efficacy.
   - Although it leverages attention to focus on relevant regions of the images, the model requires further refinement and comprehensive testing to evaluate its full potential.

### Areas for Improvement:
1. **Generalization**:
   - System 1 exhibited variability across folds, indicating challenges in ensuring consistent performance under diverse conditions.
2. **System 2 Robustness**:
   - The attention mechanism-based approach needs optimization and additional testing to unlock its potential.
3. **Configuration Testing**:
   - Comparing different model configurations (e.g., AutoEncoder setups) would provide a better understanding of their impact on performance.
4. **HoldOut Dataset Performance**:
   - The low performance on unseen data (HoldOut) suggests the need for more robust threshold tuning and model generalization strategies.

### Final Thoughts:
Both systems show potential for advancing the detection of *Helicobacter pylori* in histological images. However, further work is required to enhance robustness, refine configurations, and improve generalization, ensuring reliable performance in clinical applications.

