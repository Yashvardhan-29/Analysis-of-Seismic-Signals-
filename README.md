# Analysis-Of-Seismic-Signal
Analysis of seismic signal using Deep learning for Time series Seismic sensor data.

For identifying human activities like walking and
running, using ground vibration obtained from seismic sensors and Deep learning
algorithms. It is grounded on the idea that each activity generates distinctly unique
seismic signatures. The sampling rate of the seismic data recorded is 1000 Hz. The
datasets recorded are between 20 to 30 seconds long each. 3 and 4 channel sensors
are used to record the dataset. The data is then used to detect peaks by comparing
them with neighboring values, and peak-based segmentation is done. The segmented
time series are then labelled and used to train Deep Learning models.
Deep learning algorithms such as Recurrent Neural Networks [e.g., Long Short-Term
Memory (LSTM)] and One-Dimensional Convolutional Neural Networks, 1D CNN,
have been used for Human Activity Recognition. Human Activities recorded that
would be classified include Walking, Jogging, Hammering and Seismic Ball
Thumping.

1-D Time series data which we get will look somthing like this :
 
 ![Screenshot 2022-08-02 152414](https://user-images.githubusercontent.com/98117284/182347019-84dbe564-02e5-4d8e-8c25-c89d9ec22034.jpg)


After data processing and cleaning. I have used 1D convultional neural network algorithm to classify the Activites.

Results :

![Screenshot 2022-08-02 152748](https://user-images.githubusercontent.com/98117284/182347818-3ef57662-db3d-4133-b766-d320ccbefe8e.jpg)

![p1 (1)](https://user-images.githubusercontent.com/98117284/182347979-8eb17528-277e-4b9b-899e-890d11ddb9e8.jpg)

![Screenshot 2022-08-02 152833](https://user-images.githubusercontent.com/98117284/182348093-d26a2133-4b6a-46a8-9fa4-d814dd615e78.jpg)
