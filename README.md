Wildfire Prediction using Semi-supervised Learning

Description:

The dataset provided contains satellite images with dimensions of 350x350 pixels, categorized into two classes: "Wildfire" and "No wildfire." The original data source is Canada's Forest Fires data available on the Open Government Portal, licensed under Creative Commons 4.0 Attribution (CC-BY) license for Quebec.

The dataset consists of 22,710 images representing areas affected by wildfires and 20,140 images representing areas with no wildfire activity. To facilitate machine learning tasks, the dataset has been divided into training, testing, and validation sets, with approximately 70%, 15%, and 15% of the data allocated to each set respectively.

To construct this dataset, Longitude and Latitude coordinates corresponding to wildfire spots, where the burned area exceeds 0.01 acres, were utilized. Satellite images of these locations were extracted using the MapBox API. This extraction process aims to provide a more accessible and structured format of the dataset for deep learning purposes, facilitating the development of models capable of predicting wildfire risk in specific areas.

Advanced Neural Network Implementation:

For this task I have used the Resnet-50 as pretrained model. Here is the overview of this model.

1. Input Layer: 
   - Shape: 224x224x3 (for RGB images)
   - Typically, the input size is 224x224 pixels with 3 color channels (RGB).

2. Convolutional Layers:
   - The initial layer is a 7x7 convolutional layer with 64 filters, followed by batch normalization and ReLU activation.
   - This layer reduces the spatial dimensions of the input image.

3. Max Pooling Layer:
   - A max-pooling layer with a 3x3 kernel and a stride of 2 is applied to reduce the spatial dimensions further.

4. Residual Blocks:
   - ResNet-50 consists of 16 residual blocks grouped into convolutional blocks and identity blocks.
   - Each residual block consists of multiple convolutional layers, batch normalization, and ReLU activation functions.

5. Convolutional Blocks:
   - Convolutional blocks contain multiple convolutional layers followed by batch normalization and ReLU activation.
   - The first convolutional block after the initial layer doubles the number of filters compared to the initial convolutional layer.

6. Identity Blocks:
   - Identity blocks contain a series of convolutional layers, each followed by batch normalization and ReLU activation.
   - These blocks have a skip connection that adds the input directly to the output to preserve information flow.

7. Global Average Pooling Layer:
   - A global average pooling layer reduces the spatial dimensions to a vector by taking the average of each feature map.

8. Fully Connected Layer (Dense Layer):
   - A dense layer with a softmax activation function is added to produce the final output.
   - The number of neurons in this layer corresponds to the number of classes in the classification task.

9. Output Layer:
   - The output layer produces class probabilities for each input image.



The project integrates an Semi Supervised Pseudo Labeling  workflow to improve model performance  iteratively:

 Train Initial Model: Train a baseline model using only the labeled data. In the context of wildfire prediction, this model is a convolutional neural network (CNN) such as ResNet-50 trained on the labeled satellite images to classify them as either "wildfire" or "no wildfire."

Generate Pseudo-labels: Use the trained model to make predictions on the unlabeled data. These predictions are known as pseudo-labels. Images with high prediction confidence can be assigned pseudo-labels indicating whether they are likely to depict areas affected by wildfires or not.

Combine Labeled and Pseudo-labeled Data: Merge the labeled data with the unlabeled data that has been assigned pseudo-labels. This combined dataset is now larger and can be used for further training.

Fine-tune Model: Fine-tune the pre-trained model using the combined dataset. This involves updating the model's weights based on both the labeled data and the pseudo-labeled data. Since the pseudo-labels may contain noise, it's essential to carefully control the confidence threshold used for assigning pseudo-labels and consider using techniques to mitigate the impact of noisy labels.

Iterative Training: Optionally, repeat steps 5-7 for multiple iterations, gradually increasing the amount of labeled data through pseudo-labeling. Each iteration can potentially improve the model's performance by leveraging the unlabeled data effectively.

Evaluate Model Performance: Assess the performance of the semi-supervised model on a separate validation set or through cross-validation. This step helps determine whether the addition of pseudo-labeled data has improved the model's predictive accuracy compared to using only the initial labeled data.


By leveraging semi-supervised learning with pseudo-labeling, wildfire prediction models can make more efficient use of available data, potentially leading to improved accuracy and robustness in identifying wildfire-prone areas.
• Performance Metrics: A suite of metrics, including accuracy, precision, recall, and the F1 score, were utilized to provide a holistic evaluation of the model's performance. These metrics are particularly relevant for the imbalanced classification task at hand, offering insights into the model's ability to predict minority class instances without overwhelming false positives correctly.
