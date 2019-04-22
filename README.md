For this text classfication, i am using transfer learning method by leverage ELMo pre=-trained model which found on the TF-HUB.

### Step to test this source code

1. download sample data via https://www.kaggle.com/c/12714/download-all
2. unzip sample data and upload to the homepage of google drive
3. create python 3 notebook in COLAB
4. copy and paste to COLAB notebook
5. change runtime to GPU
6. click Ctrl + Enter to run the code
7. at the end of the running process, the submission.csv could be found on the current directory and column ['categrory'] is the prediction made by model.

### Limitation of this example code

1. ELMo is trained in English, however the sample data is in Indonesia
2, Training time in COLAB is too slow, run with juytper notebook in GCP instance will enhance the training time


