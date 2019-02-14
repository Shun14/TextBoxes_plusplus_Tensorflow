# TextBoxes++-TensorFlow
TextBoxes++ re-implementation using tensorflow.
This project is greatly inspired by [slim project](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)
And many functions are modified based on [SSD-tensorflow project](https://github.com/balancap/SSD-Tensorflow)

Author:
	Zhisheng Zou zzsshun13@gmail.com

# pretrained model 
1. [Google drive](https://drive.google.com/open?id=1kkRyVrx9iFtwEar6OJBKWNVyTLSYsF28)

# environment
` python2.7/python3.5 ` 

`tensorflow-gpu 1.8.0`

`at least one gpu`

# how to use

1. Getting the  xml file like this [example xml](./demo/example/image0.xml) and put the image together because we need the format like this [standard xml](./demo/example/standard.xml)
   1. picture format: *.png or *.PNG
2. Getting the xml and flags
   ensure the XML file is under the same directory as the corresponding image.execute the code: [convert_xml_format.py](./tools/convert_xml_format.py)
   1. `python tools/convert_xml_format.py -i in_dir  -s split_flag -l save_logs -o output_dir` 
   2. in_dir means the absolute directory which contains the pic and xml
   3. split_flag means whether or not to split the datasets
   4. save_logs means whether to save train_xml.txt
   5. output_dir means where to save xmls
3. Getting the tfrecords
   1. `python gene_tfrecords.py --xml_img_txt_path=./logs/train_xml.txt --output_dir=tfrecords` 
   2. xml_img_txt_path like this [train xml](./logs/train_xml.txt)
   3. output_dir means where to save tfrecords
4. Training
   1. `python train.py --train_dir =some_path --dataset_dir=some_path --checkpoint_path=some_path`
   2. train_dir  store the checkpoints when training
   3. dataset_dir store the tfrecords for training
   4. checkpoint_path store the model which needs to be fine tuned
5. Testing
   1. `python test.py -m /home/model.ckpt-858 -o test`
   2. -m which means the model
   3. -o which means output_result_dir
   4. -i which means the test img dir
   5. -c which means use which device to run the test
   6. -n which means the nms threshold
   7. -s which means the score threshold



# Note:

1. when you training the model, you can run the eval_result.py to eval your model and save the result
