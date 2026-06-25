# MNIST Neural Network
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) 
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) 
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
## - Carson Shae
[![email](https://img.shields.io/badge/Email-D14836?logo=gmail&logoColor=white)](mailto:carsonkshae@gmail.com) 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://linkedin.com/in/carson-shae-4a759b3b1)



## The model is trained on the fashion mnist data set from Tensor Flow:
https://www.tensorflow.org/datasets/catalog/fashion_mnist

## Training
- model uses TensorFlow Keras
- 60,000 images 
- each image is 28 pixels tall
- each image is 28 pixels wide
- 28 rows × 28 columns = 784 pixels
- Images are scaled in brigness by 0 to 255
- Dividing by 255 gives a range from 0-1
- <img width="632" height="504" alt="Screenshot 2026-06-25 134000" src="https://github.com/user-attachments/assets/393dee83-f09a-4082-890e-1155b066ddc3" />
- `fit()` trains the model 
- epochs = 10 meaning the model goes through 10 times


- Flatten layer takes 28 x 28 pixels and turns them into rows
- By doing this Dense recives a flat list of numbers
## Output
- `X_train` is the image data.
- `y_train` is the correct answer/label.
- `Dense`(10) gives 10 possible outcomes from the table
- Output is given in a list of logits largest outcome is our prediction
## Table
- 0 = T-shirt/top
- 1 = Trouser
- 2 = Pullover
- 3 = Dress
- 4 = Coat
- 5 = Sandal
- 6 = Shirt
- 7 = Sneaker
- 8 = Bag
- 9 = Ankle boot
## Other Tests
### I made some other test because I was curious on how closely the model was thinking

### `rand_test()`
1. `choice` returns a random intiger between 0 and 9999
2. this `choice` gets ran through one of the 10,000 `y_test` images
3. if it was correct move on
4. if not go to `plot_incorrect()` to see what it was thinking
### `plot_incorrect()`
1. after `rand_test()` failed to predict
2. plot the image
3. show actual vs predicted
<img width="435" height="359" alt="image" src="https://github.com/user-attachments/assets/972a9161-7c6a-43c8-a899-cf7f9f449599" />
<img width="433" height="354" alt="image" src="https://github.com/user-attachments/assets/690ed60b-b9c8-47d5-aba0-06a9c6fcddb1" />

