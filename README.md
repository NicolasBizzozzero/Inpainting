# Image inpainting via dictionary learning and sparse representation
This project aims at rebuild "damaged" pictures by learning a sparse representation of non-damaged patch of the image.


## Model
The model is composed of 3 Linear regressions (one per channel) with L1 regularization (aka Lasso).
It encodes the picture to a HSV color model, normalize its pixels between [-1, 1], and learn which sparse combination of pixels can properly rebuild the picture.


## Examples
<center>
  <img src="https://github.com/NicolasBizzozzero/Inpainting/blob/master/report/res/lena_color_512_0_1.png" alt="Example Lena 10%">
  <img src="https://github.com/NicolasBizzozzero/Inpainting/blob/master/report/res/lena_color_512_0_5.png" alt="Example Lena 50%">
  <img src="https://github.com/NicolasBizzozzero/Inpainting/blob/master/report/res/outdoor_parfait.png" alt="Example outdoor">
</center>


## TODO
* Implement a CLI.
* Find a better heuristic for patch approximation order.
* Rewrite the model in PyTorch for GPU acceleration.
* Make the linear model a parameter of the Inpainting class.


## Sources
* Bin Shen and Wei Hu and Zhang, Yimin and Zhang, Yu-Jin, Image Inpainting via Sparse Representation Proceedings of the 2009 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP ’09)
* Julien Mairal Sparse coding and Dictionnary Learning for Image Analysis INRIA Visual Recognition and Machine Learning Summer School, 2010
* A. Criminisi, P. Perez, K. Toyama Region Filling and Object Removal by Exemplar-Based Image Inpainting IEEE Transaction on Image Processing (Vol 13-9), 2004
