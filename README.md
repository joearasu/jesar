Steering Angle Prediction for Autonomous Vehicles: A Comparative Analysis of Deep Learning Architectures: 
This study presents a comprehensive comparative analysis of five recurrent neural network (RNN) architectures for steering angle prediction in autonomous vehicles using behavioral cloning. The architectures evaluated are: Standard RNN, Bidirectional RNN (BRNN), Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Encoder-Decoder RNN. Each architecture was trained on a self-driving car simulator dataset with three input features (throttle, brake, speed) and one target output (steering angle). Three hyperparameter configurations were tested per architecture, resulting in 15 distinct experiments. Results demonstrate that LSTM networks achieve superior performance with an R² score of 0.9890 and RMSE of 16.26, significantly outperforming Standard RNN (R² = 0.8712), BRNN (R² = 0.9512), GRU (R² = 0.9876), and Encoder-Decoder RNN (R² = 0.9654). 

REQUIRED LIBRARIES & VERSIONS	HARDWARE REQUIREMENTS
	
	SOFTWARE REQUIREMENTS
	
Python Version - Python 3.8.12
Tensorflow - 2.4.0
Keras - 2.4.3
Numpy - 1.19.5
Pandas - 1.1.5
scikit-learn - 0.24.1
matplotlib - 3.3.4
seaborn - 0.11.1
python-engineio - 3.13.0
python-socketio - 4.6.1
Pillow - 8.0.1
opencv-python - 4.5.1.48
	
	CPU:
	
  Intel Core i7/i9 or AMD Ryzen 7/9 (6+ cores)
  16GB+ RAM

GPU:

  NVIDIA RTX 2080 or better
  6GB+ VRAM

Storage:

  10GB+ SSD space	
  
  IDE/Editor:
  
Visual Studio Code (VSCode)
Jupyter Notebook
Google Colab
