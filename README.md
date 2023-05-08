**Problem Statement:** Classify a remote sensed image for land use land cover using Fisher's Linear Discriminant classifier. Compare the output with LDA and PCA based classifiers.

**Introduction:**

We used two algorithms, Principal Component Analysis (PCA) and Linear  Discriminant Analysis (LDA), to perform image segmentation on satellite  images. After preprocessing and reducing the dimensions of the images,  we used the reduced dimensional representation to classify the images.  We compared various classification approaches such as K-means, Support  Vector Machines (SVM), and Artificial Neural Networks (ANN). Our best  prediction results were obtained using ANN, which was further improved  by post-processing the prediction matrix with a Gaussian Blur. We  achieved an accuracy of 89.12% for the Indian Pines dataset.

**How to Run:**

Step 1: Create a python virtual environment

* `python3.10 -m venv py3`
* `source py3/bin/activate`

Step 2: Install all dependencies

* `pip install -r requirements.txt`

Step 3: Change directory to source directory

* `cd src/`

Step 4:  Add [Indian Pines](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) and [Salinas](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) data in the dataset folder.

Step 5: Run main program

* `python3.10 main.py`

  ```markdown
  -> Take `.mat` files as the input with mouse
  -> Output images will be saved in the `./output` directory
  -> If trained and predicted models will get saved in `./models` directory
  -> You need to train the model first before making any predictions. 
  -> Jupyter Notebooks are present in notebook folder.
  ```

**Algorithm:**

* [Principal Component Analysis](https://towardsdatascience.com/principal-component-analysis-for-dimensionality-reduction-115a3d157bad?gi=ca0fada5eb9b)
  * No. of components used = 16

* [Fisher's Linear Discriminant analysis](https://towardsdatascience.com/fishers-linear-discriminant-intuitively-explained-52a1ba79e1bb)
  * No. of components used = 16

* [Gaussian Blur](https://medium.com/image-vision/noise-filtering-in-digital-image-processing-d12b5266847c)
  * Ran a loop on sigmas `(0.01, 3.00, 0.01)` to find the optimal sigma


**Dataset:**

* [Indian Pines dataset](./dataset/Indian Pines)
  * [Sample](./dataset/Indian Pines/indian_pines.png)
  * [Ground Truth](./dataset/Indian Pines/indian_pines_gt.png)
  
* [Salinas](./dataset/Salinas)
  * [Sample](./dataset/Salinas/salinas.png)
  * [Ground Truth](./dataset/Salinas/salinas_gt.png)

### **Result**s:

* PCA - First 3 principal components
  * Indian Pines: [1st](./output/indian_pines/pca/pca_1.png) , [2nd](./output/indian_pines/pca/pca_2.png), [3rd](./output/indian_pines/pca/pca_3.png)
  * Salinas: [1st](./output/salinas/pca/pca_1.png) , [2nd](./output/salinas/pca/pca_2.png), [3rd](./output/salinas/pca/pca_3.png)
* LDA - First 3 components
  * Indian Pines: [1st](./output/indian_pines/lda/lda_1.png) , [2nd](./output/indian_pines/lda/lda_2.png), [3rd](./output/indian_pines/lda/lda_3.png)
  * Salinas: [1st](./output/salinas/lda/lda_1.png) , [2nd](./output/salinas/lda/lda_2.png), [3rd](./output/salinas/lda/lda_3.png)
* Predictions:
  * [PCA](./output/indian_pines/pca)
  * [LDA](./output/indian_pines/lda)

**Accuracy**:

| Accuracies                                                 | PCA     | LDA     |
| ---------------------------------------------------------- | ------- | ------- |
| `With K-means`                                             | -       | -       |
| `With SVM classifier`                                      | 64.156% | 75.412% |
| `With Artificial Neural Network (ANN)`                     | 69.26%  | 82.29%  |
| `With Artificial Neural Network (ANN) + Gauss Smoothening` | 74.00%  | 89.12%  |

We cannot compute accuracy for unsupervised approach. We can also segment without any labels for those segment. Only the classification can be acknowledged

### Results on other dataset:

Indian Pines: 89.12%
Salinas: 93.25%