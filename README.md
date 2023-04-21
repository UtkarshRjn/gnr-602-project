**Problem Statement:** Classify a remote sensed image for land use land cover using Fisher's Linear Discriminant classifier. Compare the output with LDA and PCA based classifiers.

**Introduction:**

We used two algorithms, Principal Component Analysis (PCA) and Linear  Discriminant Analysis (LDA), to perform image segmentation on satellite  images. After preprocessing and reducing the dimensions of the images,  we used the reduced dimensional representation to classify the images.  We compared various classification approaches such as K-means, Support  Vector Machines (SVM), and Artificial Neural Networks (ANN). Our best  prediction results were obtained using ANN, which was further improved  by post-processing the prediction matrix with a Gaussian Blur. We  achieved an accuracy of 89.12% for the Indian Pines dataset.

**How to Run:**

<Underway>

**Algorithm:**

* [Principal Component Analysis](https://towardsdatascience.com/principal-component-analysis-for-dimensionality-reduction-115a3d157bad?gi=ca0fada5eb9b)
* [Fisher's Linear Discriminant analysis](https://towardsdatascience.com/fishers-linear-discriminant-intuitively-explained-52a1ba79e1bb)
* [Gaussian Blur](https://medium.com/image-vision/noise-filtering-in-digital-image-processing-d12b5266847c)

**Dataset:**

* [Indian Pines dataset]()

  * [Sample]
  * [Ground Truth]

  <Can add more datasets>

**Approach:**

<Explained in Introduction try explaining it in report in detail>

### **Result**s:

* [PCA - First 3 principal components]
* [LDA - First 3 components]
* Predictions:
  * [PCA]
  * [LDA]

**Accuracy**:

| Accuracies                                               | PCA    | LDA     |
| -------------------------------------------------------- | ------ | ------- |
| With K-means                                             | -      | -       |
| With SVM classifier                                      |        | 75.412% |
| With Artificial Neural Network (ANN)                     | 69.26% | 82.29%  |
| With Artificial Neural Network (ANN) + Gauss Smoothening | 74.00% | 89.12%  |

We cannot compute accuracy for unsupervised approach. We can also segment without any labels for those segment. Only the classification can be acknowledged

**Mean Sqaure Error:**

| MSE                                                      | PCA    | LDA     |
| -------------------------------------------------------- | ------ | ------- |
| With K-means                                             | -      | -       |
| With SVM classifier                                      |        | 23.3199 |
| With Artificial Neural Network (ANN)                     | 28.419 | 17.4061 |
| With Artificial Neural Network (ANN) + Gauss Smoothening | 28.595 | 17.0469 |

We cannot compute mse for unsupervised approach. We can also segment without any labels for those segment. Only the classification can be acknowledged



### Note for Presentation

* Explain the approach with diagram, try to find images from google which could be used to depict our approach
* Can merge (keeping them side by side in presentation) few images to get a good diagram.