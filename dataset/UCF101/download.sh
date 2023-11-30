wget -v -nc --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

sudo apt-get install unrar
unrar -v -nc  x UCF101.rar ./

wget -v -nc --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip UCF101TrainTestSplits-RecognitionTask.zip -d ./
