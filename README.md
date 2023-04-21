This is our project done for the course GNR 602. Description of the project is:

Classify a remote sensed image for land use land cover using Fisher's Linear Discriminant classifier. Compare the output with LDA and PCA based classifiers.

### **Result**

**Accuracy**

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
